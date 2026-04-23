package com.healthhub;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * HealthHubApplication — Entry point.
 * @SpringBootApplication = @Configuration + @EnableAutoConfiguration + @ComponentScan
 * Spring Boot auto-configures everything (JPA, web, validation) based on classpath.
 */
@SpringBootApplication
public class HealthHubApplication {
    public static void main(String[] args) {
        SpringApplication.run(HealthHubApplication.class, args);
    }
}
