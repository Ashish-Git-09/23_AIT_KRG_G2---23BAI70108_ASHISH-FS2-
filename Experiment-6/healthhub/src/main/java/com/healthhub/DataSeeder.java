package com.healthhub;

import com.healthhub.model.Patient;
import com.healthhub.repository.PatientRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

/**
 * DataSeeder — runs at startup to insert sample patients.
 * Implements CommandLineRunner → Spring Boot calls run() after context loads.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DataSeeder implements CommandLineRunner {

    private final PatientRepository patientRepository;

    @Override
    public void run(String... args) {
        log.info("Seeding sample patient data...");

        patientRepository.save(Patient.builder()
                .name("Muskan Rathee").age(21).email("muskan@healthhub.com")
                .phone("9876543210").diagnosis("Seasonal Flu").bloodGroup("B+").build());

        patientRepository.save(Patient.builder()
                .name("Arjun Sharma").age(35).email("arjun@healthhub.com")
                .phone("9812345678").diagnosis("Hypertension").bloodGroup("O+").build());

        patientRepository.save(Patient.builder()
                .name("Priya Patel").age(28).email("priya@healthhub.com")
                .phone("9898765432").diagnosis("Type 2 Diabetes").bloodGroup("A+").build());

        log.info("Sample data seeded. Total patients: {}", patientRepository.count());
    }
}
