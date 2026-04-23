package com.healthhub.repository;

import com.healthhub.entity.Appointment;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

/**
 * AppointmentRepository — JPQL queries on the Appointment entity.
 */
@Repository
public interface AppointmentRepository extends JpaRepository<Appointment, Long> {

    // ── JPQL: appointments by status ───────────────────────────────────────
    @Query("SELECT a FROM Appointment a WHERE a.status = :status")
    List<Appointment> findByStatus(@Param("status") String status);

    // ── JPQL: appointments within a date range ─────────────────────────────
    @Query("SELECT a FROM Appointment a WHERE a.appointmentDate BETWEEN :start AND :end ORDER BY a.appointmentDate")
    List<Appointment> findByDateRange(@Param("start") LocalDate start, @Param("end") LocalDate end);

    // ── JPQL: appointments for a specific doctor ───────────────────────────
    @Query("SELECT a FROM Appointment a WHERE a.doctor.id = :doctorId ORDER BY a.appointmentDate DESC")
    List<Appointment> findByDoctorId(@Param("doctorId") Long doctorId);

    // ── JPQL: count appointments per doctor (aggregation) ─────────────────
    @Query("SELECT a.doctor.name, COUNT(a) FROM Appointment a GROUP BY a.doctor.name ORDER BY COUNT(a) DESC")
    List<Object[]> countAppointmentsPerDoctor();

    // ── Pagination on appointments ─────────────────────────────────────────
    Page<Appointment> findAll(Pageable pageable);

    // ── JPQL: appointments for patient name search ─────────────────────────
    @Query("SELECT a FROM Appointment a WHERE LOWER(a.patientName) LIKE LOWER(CONCAT('%', :name, '%'))")
    List<Appointment> searchByPatientName(@Param("name") String name);
}
