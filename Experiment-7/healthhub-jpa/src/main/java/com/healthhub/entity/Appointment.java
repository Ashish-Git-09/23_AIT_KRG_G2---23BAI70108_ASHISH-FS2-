package com.healthhub.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;

/**
 * Appointment Entity — "Many" side of One-to-Many with Doctor.
 *
 * Key JPA concepts demonstrated:
 * - @ManyToOne with FetchType.EAGER → Doctor IS loaded when Appointment is loaded
 *   (correct here — you almost always want to know which doctor owns an appointment)
 * - @JoinColumn → foreign key column in this table (doctor_id)
 * - @Index on appointment_date → speeds up date-range queries
 */
@Entity
@Table(name = "appointments",
       indexes = {
           @Index(name = "idx_appointment_date", columnList = "appointment_date"),
           @Index(name = "idx_appointment_status", columnList = "status")
       })
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString(exclude = "doctor")
public class Appointment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String patientName;

    @Column(name = "appointment_date", nullable = false)
    private LocalDate appointmentDate;

    private String status; // SCHEDULED, COMPLETED, CANCELLED

    private String notes;

    /**
     * EAGER fetch on ManyToOne side.
     * When we load an Appointment, we almost always need the Doctor info.
     * EAGER = Doctor is JOIN-fetched in the same query → no extra query.
     * Contrast: if this were LAZY, accessing .getDoctor() would fire another SELECT.
     */
    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "doctor_id", nullable = false)
    private Doctor doctor;
}
