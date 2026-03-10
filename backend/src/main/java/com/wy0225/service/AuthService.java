package com.wy0225.service;

import java.util.Map;

public interface AuthService {
    void sendCode(String email, String type);
    Map<String, Object> login(String identifier, String password);
    void register(String username, String email, String password, String code);
    void resetPassword(String email, String code, String newPassword);
    void changePassword(Long userId, String code, String oldPassword, String newPassword);
    void forceChangePassword(Long userId, String oldPassword, String newPassword);
}
