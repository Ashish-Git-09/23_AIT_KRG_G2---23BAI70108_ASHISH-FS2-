package com.healthhub.dto;

import jakarta.validation.constraints.*;
import lombok.*;

/**
 * PatientDTO — Data Transfer Object.
 * Separates internal model from API request/response.
 * Validation annotations live here so the API layer controls what's accepted.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PatientDTO {

    private Long id; // null on create, present on update/response

    @NotBlank(message = "Name is required")
    @Size(min = 2, max = 100)
    private String name;

    @NotNull(message = "Age is required")
    @Min(0) @Max(150)
    private Integer age;

    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    private String email;

    @NotBlank(message = "Phone is required")
    @Pattern(regexp = "^[0-9]{10}$", message = "Phone must be 10 digits")
    private String phone;

    @NotBlank(message = "Diagnosis is required")
    private String diagnosis;

    private String bloodGroup;
}
