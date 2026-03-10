package com.wy0225.service;

public interface EmailService {
    void sendVerificationCode(String to, String code, String purpose);
}
