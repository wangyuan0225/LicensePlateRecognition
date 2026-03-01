package com.wy0225.service;

import com.wy0225.entity.RecognitionRecord;
import com.wy0225.repository.RecognitionRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
public class AnalyzeService {

    private final RecognitionRecordRepository recordRepository;

    @Value("${app.upload.dir}")
    private String uploadDir;

    @Value("${app.algorithm.base-dir}")
    private String algorithmBaseDir;

    @Value("${app.algorithm.python-path}")
    private String pythonPath;

    @Value("${app.algorithm.script-name}")
    private String scriptName;

    @Value("${app.algorithm.result-dir}")
    private String resultDir;

    /**
     * Core recognition flow:
     * 1. Save uploaded image to upload dir
     * 2. Copy it to algorithm imgs dir (so the script can process it)
     * 3. Run Python script via ProcessBuilder
     * 4. Parse STDOUT output
     * 5. Save record to DB
     * 6. Return result map
     */
    public Map<String, Object> analyzeImage(MultipartFile file, String modelType, Long userId) throws Exception {
        // 1. Ensure per-user directories exist (use absolute paths to avoid Tomcat temp
        // dir issue)
        Path uploadPath = Paths.get(uploadDir, userId.toString()).toAbsolutePath();
        Files.createDirectories(uploadPath);
        Path resultPath = Paths.get(resultDir, userId.toString()).toAbsolutePath();
        Files.createDirectories(resultPath);

        // 2. Save uploaded file with a unique name
        String originalFilename = file.getOriginalFilename();
        String extension = "";
        if (originalFilename != null && originalFilename.contains(".")) {
            extension = originalFilename.substring(originalFilename.lastIndexOf("."));
        }
        String savedFilename = UUID.randomUUID().toString() + extension;
        Path savedFilePath = uploadPath.resolve(savedFilename);
        // Use absolute file path to prevent Spring resolving to Tomcat temp dir
        file.transferTo(savedFilePath.toAbsolutePath().toFile());
        log.info("Image saved to: {}", savedFilePath.toAbsolutePath());

        // 3. Create a temp input dir for this single image, and a temp output dir
        Path tempInputDir = Files.createTempDirectory("lpr_input_");
        Path tempOutputDir = Files.createTempDirectory("lpr_output_");
        Path tempImagePath = tempInputDir.resolve(savedFilename);
        Files.copy(savedFilePath, tempImagePath, StandardCopyOption.REPLACE_EXISTING);

        // 4. Build the ProcessBuilder command
        File algorithmDir = new File(algorithmBaseDir).getAbsoluteFile();
        File pythonExe = new File(pythonPath).getAbsoluteFile();

        // Use absolute paths
        String scriptPath = new File(algorithmDir, scriptName).getAbsolutePath();

        List<String> command = new ArrayList<>();
        command.add(pythonExe.getAbsolutePath());
        command.add(scriptPath);
        command.add("--image_path");
        command.add(tempInputDir.toAbsolutePath().toString());
        command.add("--output");
        command.add(tempOutputDir.toAbsolutePath().toString());
        command.add("--device");
        command.add("cpu");

        log.info("Executing command: {}", String.join(" ", command));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(algorithmDir);
        processBuilder.redirectErrorStream(true);

        // Set environment to ensure UTF-8
        Map<String, String> env = processBuilder.environment();
        env.put("PYTHONIOENCODING", "utf-8");

        // 5. Execute and capture output
        Process process = processBuilder.start();

        StringBuilder outputBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), "UTF-8"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                outputBuilder.append(line).append("\n");
                log.info("Python output: {}", line);
            }
        }

        int exitCode = process.waitFor();
        String output = outputBuilder.toString();
        log.info("Python process exited with code: {}", exitCode);

        if (exitCode != 0) {
            // Clean up temp dirs
            deleteDirectory(tempInputDir);
            deleteDirectory(tempOutputDir);
            throw new RuntimeException("算法引擎执行失败，退出码: " + exitCode + "\n输出: " + output);
        }

        // 6. Parse the STDOUT output
        // Format: [1/1] filename.jpg | det=1 | plates=皖1149885 绿色双层 | time=287.8ms |
        // save=path
        ParsedResult parsed = parseOutput(output);

        // 7. Copy result image from temp output dir to the actual result dir
        String resultImageFilename = savedFilename; // Use same filename
        if (parsed.saveFilename != null && !parsed.saveFilename.isEmpty()) {
            // Check if the result file exists in temp output dir
            Path tempResultFile = tempOutputDir.resolve(savedFilename);
            if (Files.exists(tempResultFile)) {
                Path finalResultPath = resultPath.resolve(savedFilename);
                Files.copy(tempResultFile, finalResultPath, StandardCopyOption.REPLACE_EXISTING);
                resultImageFilename = savedFilename;
            }
        }

        // Clean up temp dirs
        deleteDirectory(tempInputDir);
        deleteDirectory(tempOutputDir);

        // 8. Save to database
        RecognitionRecord record = new RecognitionRecord();
        record.setUserId(userId);
        record.setOriginalImage(savedFilename);
        record.setResultImage(resultImageFilename);
        record.setPlateNumber(parsed.plateNumber);
        record.setPlateColor(parsed.plateColor);
        record.setPlateType(parsed.plateType);
        record.setModelType(modelType != null ? modelType : "yolo26");
        record.setProcessingTimeMs(parsed.timeMs);
        record.setDetectCount(parsed.detectCount);
        recordRepository.save(record);

        // 9. Build response
        Map<String, Object> result = new HashMap<>();
        result.put("recordId", record.getId());
        result.put("plateNumber", parsed.plateNumber != null ? parsed.plateNumber : "-");
        result.put("plateColor", parsed.plateColor != null ? parsed.plateColor : "-");
        result.put("plateType", parsed.plateType != null ? parsed.plateType : "-");
        result.put("confidence", 0.95); // The algorithm doesn't output confidence in stdout, use default
        result.put("processingTimeMs", parsed.timeMs != null ? parsed.timeMs : 0);
        result.put("detectCount", parsed.detectCount);
        result.put("thumbnailUrl", "/static/upload/" + userId + "/" + savedFilename);
        result.put("resultImageUrl", "/static/result/" + userId + "/" + resultImageFilename);
        result.put("originalImageUrl", "/static/upload/" + userId + "/" + savedFilename);

        return result;
    }

    /**
     * Parse the Python script STDOUT.
     * Example line: [1/9] double_lv.png | det=1 | plates=皖1149885 绿色双层 |
     * time=287.8ms | save=result\double_lv.png
     */
    private ParsedResult parseOutput(String output) {
        ParsedResult result = new ParsedResult();
        String[] lines = output.split("\n");

        // Pattern for the main result line
        Pattern mainPattern = Pattern.compile(
                "\\[\\d+/\\d+\\]\\s+\\S+\\s+\\|\\s+det=(\\d+)\\s+\\|\\s+plates=(.+?)\\s+\\|\\s+time=([\\d.]+)ms\\s+\\|\\s+save=(.+)");

        for (String line : lines) {
            line = line.trim();
            Matcher matcher = mainPattern.matcher(line);
            if (matcher.find()) {
                result.detectCount = Integer.parseInt(matcher.group(1));
                String platesStr = matcher.group(2).trim();
                result.timeMs = Double.parseDouble(matcher.group(3));
                result.saveFilename = matcher.group(4).trim();

                // Parse plates string: e.g. "皖1149885 绿色双层" or "皖1149885 绿色"
                parsePlateInfo(platesStr, result);
                break; // Take the first detection line
            }
        }

        return result;
    }

    /**
     * Parse plate info from the plates field.
     * Could be: "皖1149885 绿色双层" or "皖1149885 绿色" or "-"
     */
    private void parsePlateInfo(String platesStr, ParsedResult result) {
        if (platesStr == null || platesStr.equals("-")) {
            result.plateNumber = "-";
            result.plateColor = "-";
            result.plateType = "-";
            return;
        }

        // The plates field can contain multiple plates separated by " | "
        // We take the first one
        String firstPlate = platesStr.split("\\|")[0].trim();

        // Pattern: plateNumber color[双层]
        // e.g. "皖1149885 绿色双层" -> number=皖1149885, color=绿色, type=绿色双层
        // e.g. "粤ZR066港 黑色" -> number=粤ZR066港, color=黑色
        String[] parts = firstPlate.split("\\s+", 2);
        if (parts.length >= 1) {
            result.plateNumber = parts[0];
        }
        if (parts.length >= 2) {
            String colorAndType = parts[1];
            result.plateType = colorAndType;
            // Extract just the color (remove 双层 suffix if present)
            result.plateColor = colorAndType.replace("双层", "").trim();
        }
    }

    private void deleteDirectory(Path dir) {
        try {
            Files.walk(dir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException ignored) {
                        }
                    });
        } catch (IOException ignored) {
        }
    }

    private static class ParsedResult {
        String plateNumber;
        String plateColor;
        String plateType;
        Double timeMs;
        int detectCount;
        String saveFilename;
    }
}
