package com.wy0225.controller;

import com.wy0225.common.JwtUtil;
import com.wy0225.common.Result;
import com.wy0225.service.AdminService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/admin")
@RequiredArgsConstructor
public class AdminController {

    private final AdminService adminService;
    private final JwtUtil jwtUtil;

    // A helper to ensure the user is logged in and has the ADMIN role
    private boolean isAdmin(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer "))
            return false;
        String token = authHeader.substring(7);
        if (!jwtUtil.validateToken(token))
            return false;

        // This relies on the new "role" claim added to JwtUtil
        Object role = io.jsonwebtoken.Jwts.parser()
                .verifyWith(io.jsonwebtoken.security.Keys.hmacShaKeyFor(
                        "LprVisionSecretKey2026ForJwtTokenGenerationAndValidation"
                                .getBytes(java.nio.charset.StandardCharsets.UTF_8)))
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .get("role");

        return "ADMIN".equals(role);
    }

    @GetMapping("/users")
    public Result<List<Map<String, Object>>> getAllUsers(
            @RequestHeader(value = "Authorization", required = false) String authHeader) {
        if (!isAdmin(authHeader)) {
            return Result.error(403, "无权限访问");
        }
        return Result.success(adminService.getAllUsersForDropdown());
    }

    @GetMapping("/history")
    public Result<Map<String, Object>> getAllHistory(
            @RequestParam(value = "page", defaultValue = "1") int page,
            @RequestParam(value = "size", defaultValue = "10") int size,
            @RequestParam(value = "userId", required = false) Long userId,
            @RequestParam(value = "modelType", required = false) String modelType,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        if (!isAdmin(authHeader)) {
            return Result.error(403, "无权限访问");
        }

        return Result.success(adminService.getAllHistoryWithFilters(page, size, userId, modelType));
    }

    @GetMapping("/feedback")
    public Result<List<Map<String, Object>>> getAllFeedback(
            @RequestParam(value = "userId", required = false) Long userId,
            @RequestParam(value = "modelType", required = false) String modelType,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        if (!isAdmin(authHeader)) {
            return Result.error(403, "无权限访问");
        }

        return Result.success(adminService.getAllFeedbackWithFilters(userId, modelType));
    }

    @PutMapping("/feedback/{id}/status")
    public Result<String> updateFeedbackStatus(
            @PathVariable("id") Long id,
            @RequestBody Map<String, String> body,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        if (!isAdmin(authHeader)) {
            return Result.error(403, "无权限访问");
        }

        String status = body.get("status");
        if (status == null || (!status.equals("APPROVED") && !status.equals("REJECTED"))) {
            return Result.error(400, "无效的状态值");
        }

        boolean success = adminService.updateFeedbackStatus(id, status);
        if (success) {
            return Result.success("更新状态成功");
        } else {
            return Result.error(404, "找不到该反馈记录");
        }
    }
}
