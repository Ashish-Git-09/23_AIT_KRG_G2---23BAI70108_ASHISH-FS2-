package com.healthhub.repository;

import com.healthhub.model.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

/**
 * PatientRepository — Repository Layer.
 * Extends JpaRepository → gives us CRUD for free.
 * Spring's IoC container creates and injects this bean automatically.
 */
@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {

    Optional<Patient> findByEmail(String email);

    boolean existsByEmail(String email);
}
