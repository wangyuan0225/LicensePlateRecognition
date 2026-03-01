package com.wy0225.controller;

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

    @PostMapping("/login")
    public Result<Map<String, Object>> login(@RequestBody Map<String, String> body) {
        String email = body.get("email");
        String password = body.get("password");

        if (email == null || password == null) {
            return Result.error(400, "邮箱和密码不能为空");
        }

        try {
            Map<String, Object> data = authService.login(email, password);
            return Result.success(data);
        } catch (RuntimeException e) {
            return Result.error(401, e.getMessage());
        }
    }

    @PostMapping("/register")
    public Result<Void> register(@RequestBody Map<String, String> body) {
        String username = body.get("username");
        String email = body.get("email");
        String password = body.get("password");

        if (username == null || email == null || password == null) {
            return Result.error(400, "用户名、邮箱和密码不能为空");
        }

        try {
            authService.register(username, email, password);
            return Result.created("User registered successfully");
        } catch (RuntimeException e) {
            return Result.error(400, e.getMessage());
        }
    }
}
