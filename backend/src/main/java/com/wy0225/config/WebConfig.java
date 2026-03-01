package com.wy0225.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.io.File;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Value("${app.upload.dir}")
    private String uploadDir;

    @Value("${app.algorithm.result-dir}")
    private String resultDir;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true)
                .maxAge(3600);
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // Map /static/upload/** to the upload directory
        String uploadAbsPath = new File(uploadDir).getAbsolutePath().replace("\\", "/");
        registry.addResourceHandler("/static/upload/**")
                .addResourceLocations("file:" + uploadAbsPath + "/");

        // Map /static/result/** to the algorithm result directory
        String resultAbsPath = new File(resultDir).getAbsolutePath().replace("\\", "/");
        registry.addResourceHandler("/static/result/**")
                .addResourceLocations("file:" + resultAbsPath + "/");
    }
}
