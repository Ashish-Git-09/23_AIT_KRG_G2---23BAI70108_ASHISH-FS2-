package com.healthhub.service;

import com.healthhub.dto.PatientDTO;
import com.healthhub.exception.DuplicateEmailException;
import com.healthhub.exception.ResourceNotFoundException;
import com.healthhub.model.Patient;
import com.healthhub.repository.PatientRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * PatientService — Business Logic Layer.
 * @Service tells Spring to register this as a bean (IoC).
 * @RequiredArgsConstructor + final fields = constructor-based Dependency Injection.
 * The repository is INJECTED by Spring, not created with 'new'.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PatientService {

    // Spring injects PatientRepository here via constructor injection (DI principle)
    private final PatientRepository patientRepository;

    // ── CREATE ────────────────────────────────────────────────────────────────
    public PatientDTO createPatient(PatientDTO dto) {
        log.info("Creating patient with email: {}", dto.getEmail());

        if (patientRepository.existsByEmail(dto.getEmail())) {
            throw new DuplicateEmailException("Patient with email '" + dto.getEmail() + "' already exists.");
        }

        Patient patient = toEntity(dto);
        Patient saved = patientRepository.save(patient);
        log.info("Patient created with ID: {}", saved.getId());
        return toDTO(saved);
    }

    // ── READ ALL ──────────────────────────────────────────────────────────────
    public List<PatientDTO> getAllPatients() {
        log.info("Fetching all patients");
        return patientRepository.findAll()
                .stream()
                .map(this::toDTO)
                .toList();
    }

    // ── READ ONE ──────────────────────────────────────────────────────────────
    public PatientDTO getPatientById(Long id) {
        log.info("Fetching patient with ID: {}", id);
        Patient patient = patientRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Patient not found with ID: " + id));
        return toDTO(patient);
    }

    // ── UPDATE ────────────────────────────────────────────────────────────────
    public PatientDTO updatePatient(Long id, PatientDTO dto) {
        log.info("Updating patient with ID: {}", id);
        Patient existing = patientRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Patient not found with ID: " + id));

        // Check if email is being changed to one that already exists
        if (!existing.getEmail().equals(dto.getEmail()) && patientRepository.existsByEmail(dto.getEmail())) {
            throw new DuplicateEmailException("Email '" + dto.getEmail() + "' is already in use.");
        }

        existing.setName(dto.getName());
        existing.setAge(dto.getAge());
        existing.setEmail(dto.getEmail());
        existing.setPhone(dto.getPhone());
        existing.setDiagnosis(dto.getDiagnosis());
        existing.setBloodGroup(dto.getBloodGroup());

        Patient updated = patientRepository.save(existing);
        log.info("Patient updated: {}", updated.getId());
        return toDTO(updated);
    }

    // ── DELETE ────────────────────────────────────────────────────────────────
    public void deletePatient(Long id) {
        log.info("Deleting patient with ID: {}", id);
        if (!patientRepository.existsById(id)) {
            throw new ResourceNotFoundException("Patient not found with ID: " + id);
        }
        patientRepository.deleteById(id);
        log.info("Patient deleted: {}", id);
    }

    // ── DTO Mapping ───────────────────────────────────────────────────────────
    private PatientDTO toDTO(Patient p) {
        return PatientDTO.builder()
                .id(p.getId())
                .name(p.getName())
                .age(p.getAge())
                .email(p.getEmail())
                .phone(p.getPhone())
                .diagnosis(p.getDiagnosis())
                .bloodGroup(p.getBloodGroup())
                .build();
    }

    private Patient toEntity(PatientDTO dto) {
        return Patient.builder()
                .name(dto.getName())
                .age(dto.getAge())
                .email(dto.getEmail())
                .phone(dto.getPhone())
                .diagnosis(dto.getDiagnosis())
                .bloodGroup(dto.getBloodGroup())
                .build();
    }
}
