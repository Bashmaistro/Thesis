package com.example.Thesis.Entity;


import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;


@Data
@Table(name = "Patient")
@Entity
public class Patient {

    @Id
    @Column(name = "patient_id")
    private String pid;


    @ManyToOne
    @JoinColumn(name = "doctor_id")
    private Doctor doctor;

    @Column(name = "dcm_file_path")
    private String dcmFilePath;

    @Column(name = "mri_file_path")
    private String mriFilePath;

    @Column(name = "clinic_data")
    private String clinicData;


}
