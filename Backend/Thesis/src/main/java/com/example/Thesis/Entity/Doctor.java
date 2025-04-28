package com.example.Thesis.Entity;


import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@Data
@Entity
@Table(name = "Doctor")
public class Doctor {

    @Id
    @Column(name = "doctor_id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int doctorId;


    @Column(name = "first_name")
    private String fname;

    @Column(name = "last_name")
    private String lname;

}
