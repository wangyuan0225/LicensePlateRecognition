package com.wy0225.controller;

import com.wy0225.common.JwtUtil;
import com.wy0225.common.Result;
import com.wy0225.service.HistoryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/history")
@RequiredArgsConstructor
public class HistoryController {

    private final HistoryService historyService;
    private final JwtUtil jwtUtil;

    @GetMapping("/list")
    public Result<Map<String, Object>> getHistoryList(
            @RequestParam(value = "page", defaultValue = "1") int page,
            @RequestParam(value = "size", defaultValue = "10") int size,
            @RequestParam(value = "keyword", required = false) String keyword,
            @RequestParam(value = "startDate", required = false) String startDate,
            @RequestParam(value = "endDate", required = false) String endDate,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        Map<String, Object> data = historyService.getHistoryList(userId, page, size, keyword, startDate, endDate);
        return Result.success(data);
    }

    @DeleteMapping("/{id}")
    public Result<Void> deleteRecord(@PathVariable Long id,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {
        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        try {
            historyService.deleteRecord(id);
            return Result.success("Record deleted successfully", null);
        } catch (RuntimeException e) {
            return Result.error(404, e.getMessage());
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
