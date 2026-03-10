package com.wy0225.service;

import org.springframework.web.multipart.MultipartFile;
import java.util.Map;

public interface AnalyzeService {
    Map<String, Object> analyzeImage(MultipartFile file, String modelType, Long userId) throws Exception;
}
