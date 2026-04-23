# Experiment 6 — Spring Boot REST API: HealthHub

## Aim
Design and implement a Spring Boot REST API using **Layered Architecture**, applying **IoC** and **Dependency Injection** to build a structured backend for the HealthHub application, including CRUD operations, DTO mapping, validation, and global exception handling.

---

## Architecture Overview

```
Client (Postman / Browser)
        │
        ▼
┌──────────────────┐
│  Controller Layer │  ← @RestController  (HTTP in/out, routes to service)
└────────┬─────────┘
         │  injects (DI)
┌────────▼─────────┐
│   Service Layer   │  ← @Service         (business logic, DTO mapping)
└────────┬─────────┘
         │  injects (DI)
┌────────▼─────────┐
│ Repository Layer  │  ← @Repository      (JPA CRUD, DB access)
└────────┬─────────┘
         │
┌────────▼─────────┐
│   H2 Database     │  (in-memory, auto-created)
└──────────────────┘
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---|---|
| **IoC** | Spring creates and manages all beans (`@Service`, `@Repository`, etc.) |
| **DI** | `@RequiredArgsConstructor` + `final` field = constructor injection |
| **Layered Architecture** | Controller → Service → Repository → DB |
| **DTO Mapping** | `PatientDTO` ↔ `Patient` entity — API never exposes entity directly |
| **Validation** | `@Valid` on controller + Jakarta annotations on DTO |
| **Global Exception Handling** | `@ControllerAdvice` in `GlobalExceptionHandler` |

---

## REST API Endpoints

| Method | URL | Description |
|---|---|---|
| `POST` | `/api/patients` | Create a new patient |
| `GET` | `/api/patients` | Get all patients |
| `GET` | `/api/patients/{id}` | Get patient by ID |
| `PUT` | `/api/patients/{id}` | Update patient |
| `DELETE` | `/api/patients/{id}` | Delete patient |

---

## How to Run

### Prerequisites
- Java 17+
- Maven 3.6+

### Steps
```bash
cd Experiment-6/healthhub
mvn spring-boot:run
```

Server starts at: `http://localhost:8080`
H2 Console: `http://localhost:8080/h2-console` (JDBC URL: `jdbc:h2:mem:healthhubdb`)

---

## Sample Requests (Postman / curl)

### Create Patient
```bash
curl -X POST http://localhost:8080/api/patients \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rahul Verma",
    "age": 30,
    "email": "rahul@test.com",
    "phone": "9911223344",
    "diagnosis": "Migraine",
    "bloodGroup": "AB+"
  }'
```

### Get All Patients
```bash
curl http://localhost:8080/api/patients
```

### Get Patient by ID
```bash
curl http://localhost:8080/api/patients/1
```

### Update Patient
```bash
curl -X PUT http://localhost:8080/api/patients/1 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rahul Verma Updated",
    "age": 31,
    "email": "rahul@test.com",
    "phone": "9911223344",
    "diagnosis": "Chronic Migraine",
    "bloodGroup": "AB+"
  }'
```

### Delete Patient
```bash
curl -X DELETE http://localhost:8080/api/patients/1
```

---

## Validation Error Response (example)
If you send invalid data, you get:
```json
{
  "timestamp": "2026-04-17T10:00:00",
  "status": 400,
  "error": "Validation Failed",
  "fieldErrors": {
    "email": "Invalid email format",
    "phone": "Phone must be 10 digits"
  }
}
```
