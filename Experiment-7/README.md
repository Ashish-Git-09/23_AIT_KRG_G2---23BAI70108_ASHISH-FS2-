# Experiment 7 — JPA & Database Performance Optimization

## Aim
Understand and implement JPA entity mapping, advanced JPQL querying, fetch strategies, and database performance optimization techniques using Spring Boot, JPA, and Hibernate.

---

## Key Concepts Demonstrated

### 1. JPA Entity Mapping
- `@Entity`, `@Table`, `@Id`, `@GeneratedValue`
- **One-to-Many** relationship: `Doctor` (1) → `Appointment` (N)
- `@OneToMany(mappedBy = "doctor")` on Doctor side
- `@ManyToOne` + `@JoinColumn(name = "doctor_id")` on Appointment side
- ORM maps Java objects to `doctors` and `appointments` tables automatically

### 2. JPQL (Java Persistence Query Language)
- Uses **class names and field names**, not table/column names
- `SELECT d FROM Doctor d WHERE d.specialization = :spec`
- Aggregation: `SELECT a.doctor.name, COUNT(a) FROM Appointment a GROUP BY a.doctor.name`
- JPQL is **database-agnostic** — works with any supported DB

### 3. Fetch Strategies
| Strategy | When to use | SQL behaviour |
|---|---|---|
| `FetchType.LAZY` | Collections (`@OneToMany`) | No JOIN — loads only when `.get()` is called |
| `FetchType.EAGER` | Single references (`@ManyToOne`) | Always JOINed in the main query |

### 4. N+1 Query Problem & Solution

**Problem (Naive):**
```
SELECT * FROM doctors          ← 1 query
SELECT * FROM appointments WHERE doctor_id = 1   ← query for each doctor!
SELECT * FROM appointments WHERE doctor_id = 2
... (N more queries)
= 1 + N total queries
```

**Solution (JOIN FETCH):**
```
SELECT DISTINCT d FROM Doctor d LEFT JOIN FETCH d.appointments
← 1 single query with JOIN — no N+1!
```

### 5. Database Indexing
```java
@Table(indexes = {
    @Index(name = "idx_doctor_specialization", columnList = "specialization"),
    @Index(name = "idx_appointment_date", columnList = "appointment_date")
})
```
Indexes speed up `WHERE` clauses on those columns dramatically.

### 6. Cursor-Based Pagination
```java
Pageable pageable = PageRequest.of(page, size, Sort.by("name"));
Page<Doctor> result = doctorRepository.findAll(pageable);
// Generates: SELECT * FROM doctors ORDER BY name LIMIT ? OFFSET ?
```

---

## REST Endpoints

| Method | URL | Description |
|---|---|---|
| `GET` | `/api/doctors` | All doctors |
| `GET` | `/api/doctors/specialization/{spec}` | JPQL filter by specialization |
| `GET` | `/api/doctors/search?name=...` | JPQL name search |
| `GET` | `/api/doctors/experienced?minYears=5` | JPQL experienced doctors |
| `GET` | `/api/doctors/paginated?page=0&size=3` | Paginated results |
| `GET` | `/api/doctors/with-appointments/naive` | **N+1 demo** (watch SQL logs!) |
| `GET` | `/api/doctors/with-appointments/optimized` | **JOIN FETCH** (1 query only) |
| `GET` | `/api/appointments/status/SCHEDULED` | Filter by status |
| `GET` | `/api/appointments/daterange?start=...&end=...` | Date range query |
| `GET` | `/api/appointments/count-per-doctor` | Aggregation query |
| `GET` | `/api/appointments/paginated` | Paginated appointments |

---

## How to Run

```bash
cd Experiment-7/healthhub-jpa
mvn spring-boot:run
```

**To see N+1 vs JOIN FETCH difference:**
1. Call `/api/doctors/with-appointments/naive` — watch console for multiple SQL queries
2. Call `/api/doctors/with-appointments/optimized` — only 1 SQL query fires!
