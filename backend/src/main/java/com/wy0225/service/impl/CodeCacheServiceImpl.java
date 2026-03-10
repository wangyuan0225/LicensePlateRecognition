package com.wy0225.service.impl;

import com.wy0225.service.*;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 验证码内存缓存服务（无需 Redis，适合小项目）
 * key = email + ":" + type，value = {code, expireAt}
 */
@Service
public class CodeCacheServiceImpl implements CodeCacheService {

    @Value("${app.mail.code-expire-minutes:5}")
    private int expireMinutes;

    private static final SecureRandom RANDOM = new SecureRandom();

    // key: "email:type"
    private final Map<String, CodeEntry> cache = new ConcurrentHashMap<>();

    /** 生成并缓存验证码，返回 6 位数字字符串 */
    public String generate(String email, String type) {
        String code = String.format("%06d", RANDOM.nextInt(1_000_000));
        Instant expireAt = Instant.now().plusSeconds((long) expireMinutes * 60);
        cache.put(buildKey(email, type), new CodeEntry(code, expireAt));
        return code;
    }

    /**
     * 验证验证码
     * 
     * @return true=通过，false=错误或过期
     */
    public boolean verify(String email, String code, String type) {
        String key = buildKey(email, type);
        CodeEntry entry = cache.get(key);
        if (entry == null)
            return false;
        if (Instant.now().isAfter(entry.expireAt())) {
            cache.remove(key);
            return false;
        }
        if (!entry.code().equals(code))
            return false;
        cache.remove(key); // 验证通过后立即销毁，防止重放
        return true;
    }

    private String buildKey(String email, String type) {
        return email + ":" + type;
    }

    private record CodeEntry(String code, Instant expireAt) {
    }
}
