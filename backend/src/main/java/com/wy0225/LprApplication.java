package com.wy0225;

import com.wy0225.config.AlgorithmConfig;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(AlgorithmConfig.class)
public class LprApplication {
    public static void main(String[] args) {
        SpringApplication.run(LprApplication.class, args);
    }
}
