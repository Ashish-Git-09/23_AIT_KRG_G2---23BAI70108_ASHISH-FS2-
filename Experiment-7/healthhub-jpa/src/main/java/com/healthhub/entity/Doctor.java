package com.healthhub.entity;

import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Doctor Entity — "One" side of One-to-Many relationship with Appointment.
 *
 * Key JPA concepts demonstrated:
 * - @OneToMany with mappedBy → Doctor owns the relationship description
 * - FetchType.LAZY (default for collections) → appointments NOT loaded until accessed
 * - cascade = CascadeType.ALL → deleting doctor deletes all their appointments
 * - @Index on 'specialization' → performance optimization for filtered queries
 */
@Entity
@Table(name = "doctors",
       indexes = {
           @Index(name = "idx_doctor_specialization", columnList = "specialization"),
           @Index(name = "idx_doctor_name", columnList = "name")
       })
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString(exclude = "appointments") // prevent infinite recursion in logging
public class Doctor {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String specialization;

    private String email;

    private Integer experienceYears;

    /**
     * LAZY fetch (default for @OneToMany).
     * Hibernate will NOT run a SELECT for appointments when you load a Doctor.
     * It only runs the query when you call doctor.getAppointments().
     * This is the correct default — avoids loading thousands of rows unnecessarily.
     */
    @OneToMany(mappedBy = "doctor", cascade = CascadeType.ALL, fetch = FetchType.LAZY, orphanRemoval = true)
    @Builder.Default
    private List<Appointment> appointments = new ArrayList<>();
}
