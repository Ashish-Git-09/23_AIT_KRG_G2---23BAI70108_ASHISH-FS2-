package com.healthhub;

import com.healthhub.entity.Appointment;
import com.healthhub.entity.Doctor;
import com.healthhub.repository.AppointmentRepository;
import com.healthhub.repository.DoctorRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

@SpringBootApplication
public class HealthHubJpaApplication {
    public static void main(String[] args) {
        SpringApplication.run(HealthHubJpaApplication.class, args);
    }
}

@Slf4j
@Component
@RequiredArgsConstructor
class DataSeeder implements CommandLineRunner {

    private final DoctorRepository doctorRepository;
    private final AppointmentRepository appointmentRepository;

    @Override
    public void run(String... args) {
        log.info("Seeding doctors and appointments...");

        Doctor d1 = doctorRepository.save(Doctor.builder()
                .name("Dr. Anil Kapoor").specialization("Cardiology")
                .email("anil@healthhub.com").experienceYears(12).build());

        Doctor d2 = doctorRepository.save(Doctor.builder()
                .name("Dr. Sneha Mehta").specialization("Neurology")
                .email("sneha@healthhub.com").experienceYears(8).build());

        Doctor d3 = doctorRepository.save(Doctor.builder()
                .name("Dr. Raj Kumar").specialization("Cardiology")
                .email("raj@healthhub.com").experienceYears(15).build());

        Doctor d4 = doctorRepository.save(Doctor.builder()
                .name("Dr. Priya Singh").specialization("Dermatology")
                .email("priya@healthhub.com").experienceYears(5).build());

        // Appointments for d1
        appointmentRepository.save(Appointment.builder().patientName("Muskan Rathee")
                .appointmentDate(LocalDate.now()).status("SCHEDULED").notes("Routine checkup").doctor(d1).build());
        appointmentRepository.save(Appointment.builder().patientName("Arjun Sharma")
                .appointmentDate(LocalDate.now().minusDays(2)).status("COMPLETED").notes("Follow-up").doctor(d1).build());
        appointmentRepository.save(Appointment.builder().patientName("Priya Patel")
                .appointmentDate(LocalDate.now().plusDays(1)).status("SCHEDULED").notes("ECG required").doctor(d1).build());

        // Appointments for d2
        appointmentRepository.save(Appointment.builder().patientName("Rahul Verma")
                .appointmentDate(LocalDate.now()).status("SCHEDULED").notes("MRI results review").doctor(d2).build());
        appointmentRepository.save(Appointment.builder().patientName("Kavya Nair")
                .appointmentDate(LocalDate.now().minusDays(5)).status("COMPLETED").notes("Migraine treatment").doctor(d2).build());

        // Appointments for d3
        appointmentRepository.save(Appointment.builder().patientName("Vikram Joshi")
                .appointmentDate(LocalDate.now().plusDays(3)).status("SCHEDULED").notes("Heart scan").doctor(d3).build());

        // Appointment for d4
        appointmentRepository.save(Appointment.builder().patientName("Ananya Das")
                .appointmentDate(LocalDate.now().minusDays(1)).status("CANCELLED").notes("Rescheduled").doctor(d4).build());

        log.info("Seeded {} doctors and {} appointments", doctorRepository.count(), appointmentRepository.count());
    }
}
