package com.wy0225.controller;

import com.wy0225.common.JwtUtil;
import com.wy0225.common.Result;
import com.wy0225.service.AnalyzeService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/v1/analyze")
@RequiredArgsConstructor
public class AnalyzeController {

    private final AnalyzeService analyzeService;
    private final JwtUtil jwtUtil;

    @PostMapping("/upload")
    public Result<Map<String, Object>> uploadAndAnalyze(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "modelType", required = false, defaultValue = "yolo26") String modelType,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        // Require login
        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        if (file.isEmpty()) {
            return Result.error(400, "请上传图片文件");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            return Result.error(400, "仅支持图片文件格式");
        }

        try {
            Map<String, Object> result = analyzeService.analyzeImage(file, modelType, userId);
            return Result.success("Analysis Complete", result);
        } catch (Exception e) {
            log.error("Analysis failed", e);
            return Result.error(500, "算法引擎执行失败: " + e.getMessage());
        }
    }

    private Long extractUserId(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return null;
        }
        String token = authHeader.substring(7);
        if (!jwtUtil.validateToken(token)) {
            return null;
        }
        return jwtUtil.getUserIdFromToken(token);
    }
}
