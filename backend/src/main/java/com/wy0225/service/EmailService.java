package com.wy0225.service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class EmailService {

    private final JavaMailSender mailSender;

    @Value("${app.mail.from}")
    private String from;

    /**
     * 发送 HTML 格式验证码邮件
     *
     * @param to      收件人邮箱
     * @param code    验证码
     * @param purpose 用途描述（中文），如 注册、重置密码、修改密码
     */
    public void sendVerificationCode(String to, String code, String purpose) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");

            helper.setFrom(from);
            helper.setTo(to);
            helper.setSubject("LPR Vision — 您的验证码");

            String html = buildHtml(code, purpose);
            helper.setText(html, true);

            mailSender.send(message);
        } catch (MessagingException e) {
            throw new RuntimeException("邮件发送失败，请检查邮件服务配置", e);
        }
    }

    private String buildHtml(String code, String purpose) {
        return """
                <!DOCTYPE html>
                <html lang="zh-CN">
                <head><meta charset="UTF-8"></head>
                <body style="font-family: 'Helvetica Neue', Arial, sans-serif; background:#f4f4f5; margin:0; padding:40px 0;">
                  <div style="max-width:480px; margin:0 auto; background:#fff; border-radius:12px;
                              box-shadow:0 2px 12px rgba(0,0,0,.08); overflow:hidden;">
                    <div style="background:linear-gradient(135deg,#2563eb,#7c3aed); padding:32px 40px; text-align:center;">
                      <h1 style="color:#fff; margin:0; font-size:22px; font-weight:700; letter-spacing:-.5px;">
                        LPR Vision
                      </h1>
                      <p style="color:rgba(255,255,255,.8); margin:8px 0 0; font-size:14px;">高精度车牌识别系统</p>
                    </div>
                    <div style="padding:40px;">
                      <p style="color:#374151; font-size:15px; margin:0 0 24px;">您正在进行 <strong>%s</strong> 操作，验证码为：</p>
                      <div style="background:#f3f4f6; border-radius:10px; padding:20px; text-align:center; margin:0 0 24px;">
                        <span style="font-size:36px; font-weight:700; letter-spacing:12px; color:#2563eb;">%s</span>
                      </div>
                      <p style="color:#6b7280; font-size:13px; margin:0;">验证码 <strong>5 分钟</strong>内有效，请勿泄露给他人。</p>
                    </div>
                    <div style="background:#f9fafb; border-top:1px solid #e5e7eb; padding:16px 40px; text-align:center;">
                      <p style="color:#9ca3af; font-size:12px; margin:0;">如非本人操作，请忽略此邮件。</p>
                    </div>
                  </div>
                </body>
                </html>
                """
                .formatted(purpose, code);
    }
}
