package com.wy0225.service;

import com.wy0225.config.AlgorithmConfig;
import com.wy0225.entity.RecognitionRecord;
import com.wy0225.repository.RecognitionRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
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
    private final AlgorithmConfig algorithmConfig;

    /**
     * Core recognition flow:
     * 1. Save uploaded image to per-user upload dir
     * 2. Copy to temp dir, run Python script via ProcessBuilder
     * 3. Parse STDOUT output (format differs by algorithm)
     * 4. Copy result image to per-user result dir
     * 5. Save record to DB and return result map
     */
    public Map<String, Object> analyzeImage(MultipartFile file, String modelType, Long userId) throws Exception {
        // Normalize modelType; default to yolo26
        String algo = (modelType != null && !modelType.isBlank()) ? modelType.toLowerCase() : "yolo26";

        AlgorithmConfig.AlgorithmProps props = algorithmConfig.getAlgorithms().get(algo);
        if (props == null) {
            throw new RuntimeException("不支持的算法: " + algo + "，可选: " + algorithmConfig.getAlgorithms().keySet());
        }

        String uploadDir = algorithmConfig.getUpload().getDir();
        String resultDir = algorithmConfig.getResult().getDir();

        // 1. Ensure per-user directories exist (absolute paths to avoid Tomcat temp
        // dir)
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
        file.transferTo(savedFilePath.toAbsolutePath().toFile());
        log.info("[{}] Image saved to: {}", algo, savedFilePath.toAbsolutePath());

        // 3. Create temp input/output dirs
        Path tempInputDir = Files.createTempDirectory("lpr_input_");
        Path tempOutputDir = Files.createTempDirectory("lpr_output_");
        Files.copy(savedFilePath, tempInputDir.resolve(savedFilename), StandardCopyOption.REPLACE_EXISTING);

        // 4. Build command
        File algorithmDir = new File(props.getBaseDir()).getAbsoluteFile();
        File pythonExe = new File(props.getPythonPath()).getAbsoluteFile();
        String scriptPath = new File(algorithmDir, props.getScriptName()).getAbsolutePath();

        List<String> command = new ArrayList<>();
        command.add(pythonExe.getAbsolutePath());
        command.add(scriptPath);
        command.add("--image_path");
        command.add(tempInputDir.toAbsolutePath().toString());
        command.add("--output");
        command.add(tempOutputDir.toAbsolutePath().toString());
        command.add("--device");
        command.add("cpu");

        log.info("[{}] Executing: {}", algo, String.join(" ", command));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(algorithmDir);
        processBuilder.redirectErrorStream(true);
        processBuilder.environment().put("PYTHONIOENCODING", "utf-8");

        // 5. Execute and capture output
        Process process = processBuilder.start();
        StringBuilder outputBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), "UTF-8"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                outputBuilder.append(line).append("\n");
                log.info("[{}] output: {}", algo, line);
            }
        }

        int exitCode = process.waitFor();
        String output = outputBuilder.toString();
        log.info("[{}] Process exited with code: {}", algo, exitCode);

        if (exitCode != 0) {
            deleteDirectory(tempInputDir);
            deleteDirectory(tempOutputDir);
            throw new RuntimeException("算法引擎执行失败，退出码: " + exitCode + "\n输出: " + output);
        }

        // 6. Parse STDOUT (different format per algorithm)
        ParsedResult parsed = "yolov8".equals(algo)
                ? parseYolov8Output(output)
                : parseYolo26Output(output);

        // 7. Copy result image from temp output to actual result dir
        String resultImageFilename = savedFilename;
        Path tempResultFile = tempOutputDir.resolve(savedFilename);
        if (Files.exists(tempResultFile)) {
            Path finalResultPath = resultPath.resolve(savedFilename);
            Files.copy(tempResultFile, finalResultPath, StandardCopyOption.REPLACE_EXISTING);
        }

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
        record.setModelType(algo);
        record.setProcessingTimeMs(parsed.timeMs);
        record.setDetectCount(parsed.detectCount);
        recordRepository.save(record);

        // 9. Build response
        Map<String, Object> result = new HashMap<>();
        result.put("recordId", record.getId());
        result.put("plateNumber", parsed.plateNumber != null ? parsed.plateNumber : "-");
        result.put("plateColor", parsed.plateColor != null ? parsed.plateColor : "-");
        result.put("plateType", parsed.plateType != null ? parsed.plateType : "-");
        result.put("modelType", algo);
        result.put("confidence", 0.95);
        result.put("processingTimeMs", parsed.timeMs != null ? parsed.timeMs : 0);
        result.put("detectCount", parsed.detectCount);
        result.put("thumbnailUrl", "/static/upload/" + userId + "/" + savedFilename);
        result.put("resultImageUrl", "/static/result/" + userId + "/" + resultImageFilename);
        result.put("originalImageUrl", "/static/upload/" + userId + "/" + savedFilename);

        return result;
    }

    /**
     * Parse yolo26 STDOUT.
     * Example: [1/9] double_lv.png | det=1 | plates=皖1149885 绿色双层 | time=287.8ms |
     * save=result\xxx.png
     */
    private ParsedResult parseYolo26Output(String output) {
        ParsedResult result = new ParsedResult();
        Pattern pattern = Pattern.compile(
                "\\[\\d+/\\d+\\]\\s+\\S+\\s+\\|\\s+det=(\\d+)\\s+\\|\\s+plates=(.+?)\\s+\\|\\s+time=([\\d.]+)ms\\s+\\|\\s+save=(.+)");

        for (String line : output.split("\n")) {
            Matcher m = pattern.matcher(line.trim());
            if (m.find()) {
                result.detectCount = Integer.parseInt(m.group(1));
                result.timeMs = Double.parseDouble(m.group(3));
                result.saveFilename = m.group(4).trim();
                parsePlateInfo(m.group(2).trim(), result);
                break;
            }
        }
        return result;
    }

    /**
     * Parse yolov8 STDOUT.
     * The script calls draw_result which does: print(result_str)
     * result_str format: "皖1149885 绿色 " or "皖A12345 蓝色双层 "
     * Also prints timing: "sumTime time is X s, average pic time is Y"
     */
    private ParsedResult parseYolov8Output(String output) {
        ParsedResult result = new ParsedResult();
        String[] lines = output.split("\n");

        // Timing line: "sumTime time is X s, average pic time is Y"
        Pattern timePattern = Pattern.compile("sumTime time is ([\\d.]+) s");

        for (String line : lines) {
            line = line.trim();

            // Timing
            Matcher tm = timePattern.matcher(line);
            if (tm.find()) {
                result.timeMs = Double.parseDouble(tm.group(1)) * 1000;
                continue;
            }

            // The plate result line: printed by draw_result's print(result_str)
            // Looks like: "皖1149885 绿色 " or "皖A12345 黄色双层 粤B12345 蓝色 "
            // It won't match the [x/x] pattern or model param line
            if (!line.isEmpty()
                    && !line.startsWith("[")
                    && !line.contains("params")
                    && !line.contains("sumTime")
                    && !line.contains("\\")
                    && !line.contains("/")
                    && !line.matches("\\d+.*")) {

                // Split multiple plates by 2+ spaces
                String[] plateTokens = line.trim().split("\\s{2,}");
                if (plateTokens.length > 0 && !plateTokens[0].isBlank()) {
                    parsePlateInfo(plateTokens[0].trim(), result);
                    result.detectCount++;
                }
            }
        }
        return result;
    }

    /**
     * Parse plate info from plates field.
     * e.g. "皖1149885 绿色双层" → number=皖1149885, color=绿色, type=绿色双层
     */
    private void parsePlateInfo(String platesStr, ParsedResult result) {
        if (platesStr == null || platesStr.equals("-") || platesStr.isBlank()) {
            result.plateNumber = "-";
            result.plateColor = "-";
            result.plateType = "-";
            return;
        }

        String firstPlate = platesStr.split("\\|")[0].trim();
        String[] parts = firstPlate.split("\\s+", 2);
        if (parts.length >= 1) {
            result.plateNumber = parts[0];
        }
        if (parts.length >= 2) {
            String colorAndType = parts[1].trim();
            result.plateType = colorAndType;
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
