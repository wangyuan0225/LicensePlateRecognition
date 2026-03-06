package com.wy0225.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.Map;

@Data
@ConfigurationProperties(prefix = "app")
public class AlgorithmConfig {

    private Map<String, AlgorithmProps> algorithms;
    private UploadProps upload;
    private ResultProps result;

    @Data
    public static class AlgorithmProps {
        private String baseDir;
        private String pythonPath;
        private String scriptName;
        /** Optional: detect model path, relative to baseDir (used by yolov8) */
        private String detectModel;
        /** Optional: recognition model path, relative to baseDir (used by yolov8) */
        private String recModel;
    }

    @Data
    public static class UploadProps {
        private String dir;
    }

    @Data
    public static class ResultProps {
        private String dir;
    }
}
