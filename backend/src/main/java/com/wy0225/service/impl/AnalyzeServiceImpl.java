package com.wy0225.service.impl;

import com.wy0225.service.*;

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
public class AnalyzeServiceImpl implements AnalyzeService {

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
        long startTime = System.currentTimeMillis();
        String algo = (modelType != null && !modelType.isBlank()) ? modelType.toLowerCase() : "yolo26";

        String uploadDir = algorithmConfig.getUpload().getDir();
        String resultDir = algorithmConfig.getResult().getDir();

        Path uploadPath = Paths.get(uploadDir, userId.toString()).toAbsolutePath();
        Files.createDirectories(uploadPath);
        Path resultPath = Paths.get(resultDir, userId.toString()).toAbsolutePath();
        Files.createDirectories(resultPath);

        String originalFilename = file.getOriginalFilename();
        String extension = "";
        if (originalFilename != null && originalFilename.contains(".")) {
            extension = originalFilename.substring(originalFilename.lastIndexOf("."));
        }
        String savedFilename = UUID.randomUUID().toString() + extension;
        Path savedFilePath = uploadPath.resolve(savedFilename);
        file.transferTo(savedFilePath.toAbsolutePath().toFile());
        log.info("[{}] Image saved to: {}", algo, savedFilePath.toAbsolutePath());

        if ("fusion".equals(algo)) {
            return runFusionModel(algo, savedFilePath, savedFilename, resultPath, userId, startTime);
        } else {
            return runSingleModel(algo, savedFilePath, savedFilename, resultPath, userId, startTime);
        }
    }

    private Map<String, Object> runFusionModel(String metaAlgo, Path savedFilePath, String savedFilename,
            Path resultPath, Long userId, long startTime) throws Exception {
        Path tempInputDir = Files.createTempDirectory("lpr_input_");
        Files.copy(savedFilePath, tempInputDir.resolve(savedFilename), StandardCopyOption.REPLACE_EXISTING);

        List<String> algosToRun = Arrays.asList("hyperlpr", "yolov8", "yolo26", "yolov11");
        List<ParsedResult> results = new ArrayList<>();

        ParsedResult yolo26Result = null;

        for (String algo : algosToRun) {
            Path tempOutputDir = Files.createTempDirectory("lpr_output_" + algo + "_");
            try {
                ParsedResult res = executeAlgorithm(algo, tempInputDir, tempOutputDir, savedFilename);
                results.add(res);
                if ("yolo26".equals(algo)) {
                    yolo26Result = res;
                    Path tempResultFile = tempOutputDir.resolve(savedFilename);
                    if (Files.exists(tempResultFile)) {
                        Path finalResultPath = resultPath.resolve(savedFilename);
                        Files.copy(tempResultFile, finalResultPath, StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            } catch (Exception e) {
                log.error("Fusion sub-model {} failed", algo, e);
            } finally {
                deleteDirectory(tempOutputDir);
            }
        }

        deleteDirectory(tempInputDir);

        if (results.isEmpty() || yolo26Result == null) {
            throw new RuntimeException("融合算法执行失败：所有子模型均失败或主干网络失效");
        }

        Map<String, Integer> plateVotes = new HashMap<>();
        Map<String, Integer> colorVotes = new HashMap<>();

        for (ParsedResult r : results) {
            if (r.plateNumber != null && !"-".equals(r.plateNumber)) {
                plateVotes.put(r.plateNumber, plateVotes.getOrDefault(r.plateNumber, 0) + 1);
            }
            if (r.plateColor != null && !"-".equals(r.plateColor)) {
                String color = normalizeColor(r.plateColor);
                colorVotes.put(color, colorVotes.getOrDefault(color, 0) + 1);
            }
        }

        String fallbackPlate = yolo26Result.plateNumber != null ? yolo26Result.plateNumber : "-";
        String fallbackColor = yolo26Result.plateColor != null ? normalizeColor(yolo26Result.plateColor) : "-";

        String bestPlate = getBestVote(plateVotes, fallbackPlate);
        String bestColor = getBestVote(colorVotes, fallbackColor);
        if (bestPlate == null || bestPlate.isBlank())
            bestPlate = "-";
        if (bestColor == null || bestColor.isBlank())
            bestColor = "-";

        double maxTime = (double) (System.currentTimeMillis() - startTime);
        int maxDet = results.stream().mapToInt(r -> r.detectCount).max().orElse(0);

        RecognitionRecord record = new RecognitionRecord();
        record.setUserId(userId);
        record.setOriginalImage(savedFilename);
        record.setResultImage(savedFilename);
        record.setPlateNumber(bestPlate);
        record.setPlateColor(bestColor);
        record.setPlateType(bestColor);
        record.setModelType(metaAlgo);
        record.setProcessingTimeMs(maxTime);
        record.setDetectCount(maxDet);
        recordRepository.save(record);

        return buildResponseMap(record, savedFilename, savedFilename, metaAlgo, userId, maxTime, maxDet);
    }

    private String normalizeColor(String rawColor) {
        if (rawColor == null)
            return "-";
        if (rawColor.contains("蓝"))
            return "蓝色";
        if (rawColor.contains("黄"))
            return "黄色";
        if (rawColor.contains("绿"))
            return "绿色";
        if (rawColor.contains("白"))
            return "白色";
        if (rawColor.contains("黑"))
            return "黑色";
        return rawColor;
    }

    private String getBestVote(Map<String, Integer> votes, String fallback) {
        if (votes.isEmpty())
            return fallback;
        int max = 0;
        String best = fallback;
        for (Map.Entry<String, Integer> e : votes.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    private Map<String, Object> runSingleModel(String algo, Path savedFilePath, String savedFilename, Path resultPath,
            Long userId, long startTime) throws Exception {
        Path tempInputDir = Files.createTempDirectory("lpr_input_");
        Path tempOutputDir = Files.createTempDirectory("lpr_output_");
        Files.copy(savedFilePath, tempInputDir.resolve(savedFilename), StandardCopyOption.REPLACE_EXISTING);

        ParsedResult parsed;
        try {
            parsed = executeAlgorithm(algo, tempInputDir, tempOutputDir, savedFilename);
            Path tempResultFile = tempOutputDir.resolve(savedFilename);
            if (Files.exists(tempResultFile)) {
                Path finalResultPath = resultPath.resolve(savedFilename);
                Files.copy(tempResultFile, finalResultPath, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            deleteDirectory(tempInputDir);
            deleteDirectory(tempOutputDir);
        }

        double totalTime = (double) (System.currentTimeMillis() - startTime);

        RecognitionRecord record = new RecognitionRecord();
        record.setUserId(userId);
        record.setOriginalImage(savedFilename);
        record.setResultImage(savedFilename);
        record.setPlateNumber(parsed.plateNumber);
        record.setPlateColor(parsed.plateColor);
        record.setPlateType(parsed.plateType);
        record.setModelType(algo);
        record.setProcessingTimeMs(totalTime);
        record.setDetectCount(parsed.detectCount);
        recordRepository.save(record);

        return buildResponseMap(record, savedFilename, savedFilename, algo, userId, totalTime, parsed.detectCount);
    }

    private ParsedResult executeAlgorithm(String algo, Path tempInputDir, Path tempOutputDir, String filename)
            throws Exception {
        AlgorithmConfig.AlgorithmProps props = algorithmConfig.getAlgorithms().get(algo);
        if (props == null) {
            throw new RuntimeException("不支持的算法: " + algo);
        }

        File algorithmDir = new File(props.getBaseDir()).getAbsoluteFile();
        File pythonExe = new File(props.getPythonPath()).getAbsoluteFile();
        String scriptPath = new File(algorithmDir, props.getScriptName()).getAbsolutePath();

        List<String> command = new ArrayList<>();
        command.add(pythonExe.getAbsolutePath());
        command.add(scriptPath);

        if ("yolov8".equals(algo)) {
            command.add("--detect_model");
            command.add(props.getDetectModel() != null ? props.getDetectModel() : "weights/yolov8s.pt");
            command.add("--rec_model");
            command.add(props.getRecModel() != null ? props.getRecModel() : "weights/plate_rec_color.pth");
            command.add("--image_path");
            command.add(tempInputDir.toAbsolutePath().toString());
            command.add("--output");
            command.add(tempOutputDir.toAbsolutePath().toString());
        } else {
            command.add("--image_path");
            command.add(tempInputDir.toAbsolutePath().toString());
            command.add("--output");
            command.add(tempOutputDir.toAbsolutePath().toString());
            command.add("--device");
            command.add("cpu");
        }

        log.info("[{}] Executing: {}", algo, String.join(" ", command));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(algorithmDir);
        processBuilder.redirectErrorStream(true);
        processBuilder.environment().put("PYTHONIOENCODING", "utf-8");

        Process process = processBuilder.start();
        StringBuilder outputBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                outputBuilder.append(line).append("\n");
                log.info("[{}] output: {}", algo, line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("算法引擎执行失败，退出码: " + exitCode + "\n输出: " + outputBuilder.toString());
        }

        if ("yolov8".equals(algo)) {
            return parseYolov8Output(outputBuilder.toString());
        } else {
            return parseYolo26Output(outputBuilder.toString());
        }
    }

    private Map<String, Object> buildResponseMap(RecognitionRecord record, String originalImage, String resultImage,
            String algo, Long userId, Double processingTime, int detectCount) {
        Map<String, Object> result = new HashMap<>();
        result.put("recordId", record.getId());
        result.put("plateNumber", record.getPlateNumber() != null ? record.getPlateNumber() : "-");
        result.put("plateColor", record.getPlateColor() != null ? record.getPlateColor() : "-");
        result.put("plateType", record.getPlateType() != null ? record.getPlateType() : "-");
        result.put("modelType", algo);
        result.put("confidence", 0.95);
        result.put("processingTimeMs", processingTime != null ? processingTime : 0);
        result.put("detectCount", detectCount);
        result.put("thumbnailUrl", "/static/upload/" + userId + "/" + originalImage);
        result.put("resultImageUrl", "/static/result/" + userId + "/" + resultImage);
        result.put("originalImageUrl", "/static/upload/" + userId + "/" + originalImage);
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
                parsePlateInfo(m.group(2).trim(), result);
                break;
            }
        }
        return result;
    }

    /**
     * Parse yolov8 STDOUT.
     *
     * The script prints: print(count, pic_, end=" ") then draw_result calls
     * print(result_str)
     * Because end=" " has no newline, both end up on ONE line, e.g.:
     * 0 C:\...\file.png 皖1149885 绿色双层
     *
     * Strategy: use regex to scan every line for the Chinese plate pattern
     * directly,
     * rather than relying on line filtering (which was storing the full path as
     * plateColor).
     *
     * Plate pattern: Chinese province char + 5-8 alphanumeric/dot chars,
     * then a space, then 2-4 Chinese chars (color + optional 双层).
     */
    private ParsedResult parseYolov8Output(String output) {
        ParsedResult result = new ParsedResult();

        // Pattern: Chinese char (province) + alphanumeric plate body + space + Chinese
        // color
        // e.g. 皖1149885 绿色双层 or 粤ZR066港 黑色
        Pattern platePattern = Pattern.compile(
                "([\u4e00-\u9fa5][A-Z0-9·\\.]{4,8})\\s+([\u4e00-\u9fa5]{1,4}(?:双层)?)");

        // Timing line: "sumTime time is X s, average pic time is Y"
        for (String line : output.split("\\n")) {
            line = line.trim();

            if (result.plateNumber == null) {
                Matcher pm = platePattern.matcher(line);
                if (pm.find()) {
                    String plateNum = pm.group(1);
                    String colorType = pm.group(2).trim();
                    result.plateNumber = plateNum;
                    result.plateType = colorType;
                    result.plateColor = colorType.replace("双层", "").trim();
                    result.detectCount = 1;
                    log.info("[yolov8] Parsed plate: {} / {}", plateNum, colorType);
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
        int detectCount;
    }
}
