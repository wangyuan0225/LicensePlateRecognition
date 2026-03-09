package com.wy0225.controller;

import com.wy0225.common.JwtUtil;
import com.wy0225.common.Result;
import com.wy0225.service.AuthService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;
    private final JwtUtil jwtUtil;

    // ----------------------------------------------------------------
    // 发送验证码
    // POST /api/v1/auth/send-code
    // Body: { "email": "...", "type": "register|reset|change" }
    // ----------------------------------------------------------------
    @PostMapping("/send-code")
    public Result<Void> sendCode(@RequestBody Map<String, String> body) {
        String email = body.get("email");
        String type = body.get("type");

        if (email != null)
            email = email.trim();

        if (email == null || email.isBlank()) {
            return Result.error(400, "邮箱不能为空");
        }
        if (type == null || !type.matches("register|reset|change")) {
            return Result.error(400, "无效的验证码类型");
        }

        try {
            authService.sendCode(email, type);
            return Result.success(null);
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    // 登录（支持用户名 OR 邮箱）
    // POST /api/v1/auth/login
    // Body: { "identifier": "...", "password": "..." }
    // ----------------------------------------------------------------
    @PostMapping("/login")
    public Result<Map<String, Object>> login(@RequestBody Map<String, String> body) {
        String identifier = body.get("identifier");
        String password = body.get("password");

        if (identifier == null || password == null) {
            return Result.error(400, "账号和密码不能为空");
        }

        try {
            Map<String, Object> data = authService.login(identifier, password);
            return Result.success(data);
        } catch (RuntimeException e) {
            return Result.error(401, e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    // 注册（含验证码）
    // POST /api/v1/auth/register
    // Body: { "username": "...", "email": "...", "password": "...", "code": "..." }
    // ----------------------------------------------------------------
    @PostMapping("/register")
    public Result<Void> register(@RequestBody Map<String, String> body) {
        String username = body.get("username");
        String email = body.get("email");
        String password = body.get("password");
        String code = body.get("code");

        if (email != null)
            email = email.trim();
        if (code != null)
            code = code.trim();

        if (username == null || email == null || password == null || code == null) {
            return Result.error(400, "所有字段均不能为空");
        }

        try {
            authService.register(username, email, password, code);
            return Result.created("注册成功");
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    // 忘记密码（邮箱 + 验证码 + 新密码）
    // POST /api/v1/auth/forgot-password
    // Body: { "email": "...", "code": "...", "newPassword": "..." }
    // ----------------------------------------------------------------
    @PostMapping("/forgot-password")
    public Result<Void> forgotPassword(@RequestBody Map<String, String> body) {
        String email = body.get("email");
        String code = body.get("code");
        String newPassword = body.get("newPassword");

        if (email == null || code == null || newPassword == null) {
            return Result.error(400, "邮箱、验证码和新密码不能为空");
        }

        try {
            authService.resetPassword(email, code, newPassword);
            return Result.success(null);
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    // 修改密码（需 JWT，邮箱验证码 + 旧密码 + 新密码）
    // POST /api/v1/auth/change-password
    // Header: Authorization: Bearer <token>
    // Body: { "code": "...", "oldPassword": "...", "newPassword": "..." }
    // ----------------------------------------------------------------
    @PostMapping("/change-password")
    public Result<Void> changePassword(
            @RequestHeader("Authorization") String authHeader,
            @RequestBody Map<String, String> body) {

        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return Result.error(401, "未授权");
        }

        String token = authHeader.substring(7);
        Long userId;
        try {
            userId = jwtUtil.getUserIdFromToken(token);
        } catch (Exception e) {
            return Result.error(401, "Token 无效或已过期");
        }

        String code = body.get("code");
        String oldPassword = body.get("oldPassword");
        String newPassword = body.get("newPassword");

        if (code == null || oldPassword == null || newPassword == null) {
            return Result.error(400, "验证码、旧密码和新密码不能为空");
        }

        try {
            authService.changePassword(userId, code, oldPassword, newPassword);
            return Result.success(null);
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }

    @PostMapping("/force-change-password")
    public Result<Void> forceChangePassword(
            @RequestHeader("Authorization") String authHeader,
            @RequestBody Map<String, String> body) {

        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return Result.error(401, "未授权");
        }

        String token = authHeader.substring(7);
        Long userId;
        try {
            userId = jwtUtil.getUserIdFromToken(token);
        } catch (Exception e) {
            return Result.error(401, "Token 无效或已过期");
        }

        String oldPassword = body.get("oldPassword");
        String newPassword = body.get("newPassword");

        if (oldPassword == null || newPassword == null) {
            return Result.error(400, "当前密码和新密码不能为空");
        }

        try {
            authService.forceChangePassword(userId, oldPassword, newPassword);
            return Result.success(null);
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }
}
