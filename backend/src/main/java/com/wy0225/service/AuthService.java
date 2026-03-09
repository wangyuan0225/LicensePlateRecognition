package com.wy0225.service;

import com.wy0225.common.JwtUtil;
import com.wy0225.entity.User;
import com.wy0225.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final JwtUtil jwtUtil;
    private final CodeCacheService codeCacheService;
    private final EmailService emailService;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    // ----------------------------------------------------------------
    // 发送验证码
    // ----------------------------------------------------------------

    /**
     * 向指定邮箱发送验证码
     *
     * @param email 收件邮箱
     * @param type  用途：register / reset / change
     */
    public void sendCode(String email, String type) {
        // change 场景：邮箱必须已注册
        if ("change".equals(type) && !userRepository.existsByEmail(email)) {
            throw new RuntimeException("该邮箱未注册");
        }
        // reset 场景：邮箱必须已注册
        if ("reset".equals(type) && !userRepository.existsByEmail(email)) {
            throw new RuntimeException("该邮箱未注册");
        }
        // register 场景：邮箱不能已注册
        if ("register".equals(type) && userRepository.existsByEmail(email)) {
            throw new RuntimeException("该邮箱已被注册");
        }

        String code = codeCacheService.generate(email, type);
        String purpose = switch (type) {
            case "register" -> "注册";
            case "reset" -> "重置密码";
            case "change" -> "修改密码";
            default -> "身份验证";
        };
        emailService.sendVerificationCode(email, code, purpose);
    }

    // ----------------------------------------------------------------
    // 登录（支持用户名 OR 邮箱）
    // ----------------------------------------------------------------

    public Map<String, Object> login(String identifier, String password) {
        // 先尝试邮箱匹配，再尝试用户名匹配
        Optional<User> userOpt = userRepository.findByUsernameOrEmail(identifier, identifier);
        if (userOpt.isEmpty()) {
            throw new RuntimeException("用户不存在");
        }

        User user = userOpt.get();
        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new RuntimeException("密码错误");
        }

        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getRole());

        Map<String, Object> result = new HashMap<>();
        result.put("token", token);

        Map<String, Object> userInfo = new HashMap<>();
        userInfo.put("id", user.getId());
        userInfo.put("username", user.getUsername());
        userInfo.put("email", user.getEmail());
        userInfo.put("role", user.getRole());
        userInfo.put("forceChangePassword", user.getForceChangePassword());
        result.put("user", userInfo);

        return result;
    }

    // ----------------------------------------------------------------
    // 注册（含验证码校验）
    // ----------------------------------------------------------------

    public void register(String username, String email, String password, String code) {
        if (!codeCacheService.verify(email, code, "register")) {
            throw new RuntimeException("验证码错误或已过期");
        }
        if (userRepository.existsByEmail(email)) {
            throw new RuntimeException("该邮箱已被注册");
        }
        if (userRepository.existsByUsername(username)) {
            throw new RuntimeException("该用户名已被使用");
        }

        User user = new User();
        user.setUsername(username);
        user.setEmail(email);
        user.setPassword(passwordEncoder.encode(password));
        userRepository.save(user);
    }

    // ----------------------------------------------------------------
    // 忘记密码（邮箱 + 验证码 + 新密码）
    // ----------------------------------------------------------------

    public void resetPassword(String email, String code, String newPassword) {
        if (!codeCacheService.verify(email, code, "reset")) {
            throw new RuntimeException("验证码错误或已过期");
        }
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("该邮箱未注册"));
        user.setPassword(passwordEncoder.encode(newPassword));
        userRepository.save(user);
    }

    // ----------------------------------------------------------------
    // 修改密码（邮箱 + 验证码 + 旧密码 + 新密码）
    // ----------------------------------------------------------------

    public void changePassword(Long userId, String code, String oldPassword, String newPassword) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("用户不存在"));

        if (!codeCacheService.verify(user.getEmail(), code, "change")) {
            throw new RuntimeException("验证码错误或已过期");
        }
        if (!passwordEncoder.matches(oldPassword, user.getPassword())) {
            throw new RuntimeException("旧密码错误");
        }

        user.setPassword(passwordEncoder.encode(newPassword));
        userRepository.save(user);
    }

    // ----------------------------------------------------------------
    // 强制修改密码 (首次登录)
    // ----------------------------------------------------------------

    public void forceChangePassword(Long userId, String oldPassword, String newPassword) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("用户不存在"));

        if (!passwordEncoder.matches(oldPassword, user.getPassword())) {
            throw new RuntimeException("当前密码错误");
        }

        user.setPassword(passwordEncoder.encode(newPassword));
        user.setForceChangePassword(false); // 重置标记
        userRepository.save(user);
    }
}
