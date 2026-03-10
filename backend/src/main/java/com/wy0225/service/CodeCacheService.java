package com.wy0225.service;

public interface CodeCacheService {
    String generate(String email, String type);
    boolean verify(String email, String code, String type);
}
