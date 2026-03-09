-- ============================================
-- 车牌识别系统 MySQL 建表语句
-- 数据库名: lpr_db
-- ============================================

CREATE DATABASE IF NOT EXISTS lpr_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE lpr_db;

-- --------------------------------------------
-- 用户表
-- --------------------------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
    `id`         BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `username`   VARCHAR(50)  NOT NULL COMMENT '用户名',
    `email`      VARCHAR(100) NOT NULL COMMENT '邮箱',
    `password`   VARCHAR(255) NOT NULL COMMENT '密码(BCrypt加密)',
    `role`       VARCHAR(20)  DEFAULT 'USER' COMMENT '角色',
    `force_change_password` BOOLEAN DEFAULT FALSE COMMENT '强制修改密码',
    `created_at` DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`),
    UNIQUE KEY `uk_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 插入默认管理员账号
-- 默认初始密码 123456
INSERT IGNORE INTO `users` (`username`, `email`, `password`, `role`, `force_change_password`) VALUES 
('admin', 'admin@lpr.com', '$2a$10$qyl84FogPwQ/1gbH/mKXgu4N5n5TTwzWi8Yzky.ZzKbC/bxvc6Eza', 'ADMIN', TRUE);

-- --------------------------------------------
-- 识别记录表
-- --------------------------------------------
DROP TABLE IF EXISTS `recognition_records`;
CREATE TABLE `recognition_records` (
    `id`                 BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `user_id`            BIGINT       DEFAULT NULL COMMENT '操作用户ID',
    `original_image`     VARCHAR(500) DEFAULT NULL COMMENT '原始上传图片文件名',
    `result_image`       VARCHAR(500) DEFAULT NULL COMMENT '算法结果图片文件名',
    `plate_number`       VARCHAR(50)  DEFAULT NULL COMMENT '车牌号码，如 皖1149885',
    `plate_color`        VARCHAR(20)  DEFAULT NULL COMMENT '车牌颜色，如 绿色',
    `plate_type`         VARCHAR(50)  DEFAULT NULL COMMENT '车牌属性，如 绿色双层',
    `model_type`         VARCHAR(50)  DEFAULT NULL COMMENT '算法模型，如 yolo26',
    `processing_time_ms` DOUBLE       DEFAULT NULL COMMENT '识别耗时(毫秒)',
    `detect_count`       INT          DEFAULT NULL COMMENT '检测到的车牌数量',
    `created_at`         DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_user_id` (`user_id`),
    KEY `idx_plate_number` (`plate_number`),
    KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='车牌识别记录表';

-- --------------------------------------------
-- 错误反馈表
-- --------------------------------------------
DROP TABLE IF EXISTS `feedbacks`;
CREATE TABLE `feedbacks` (
    `id`                 BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `user_id`            BIGINT       DEFAULT NULL COMMENT '反馈用户ID',
    `original_image_url` VARCHAR(500) DEFAULT NULL COMMENT '原始上传图片文件名/URL',
    `result_image_url`   VARCHAR(500) DEFAULT NULL COMMENT '算法结果图片文件名/URL',
    `recognized_plate`   VARCHAR(50)  DEFAULT NULL COMMENT '算法识别的车牌号码',
    `corrected_plate`    VARCHAR(50)  DEFAULT NULL COMMENT '用户更正的车牌号码',
    `model_type`         VARCHAR(50)  DEFAULT NULL COMMENT '算法模型',
    `status`             VARCHAR(20)  DEFAULT 'PENDING' COMMENT '审批状态',
    `created_at`         DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '反馈时间',
    PRIMARY KEY (`id`),
    KEY `idx_user_id` (`user_id`),
    KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='识别错误反馈表';
