package com.wy0225.config;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.io.File;

@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {

    private final AlgorithmConfig algorithmConfig;

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
        String uploadAbsPath = new File(algorithmConfig.getUpload().getDir())
                .getAbsolutePath().replace("\\", "/");
        registry.addResourceHandler("/static/upload/**")
                .addResourceLocations("file:" + uploadAbsPath + "/");

        // Map /static/result/** to the result directory
        String resultAbsPath = new File(algorithmConfig.getResult().getDir())
                .getAbsolutePath().replace("\\", "/");
        registry.addResourceHandler("/static/result/**")
                .addResourceLocations("file:" + resultAbsPath + "/");
    }
}
