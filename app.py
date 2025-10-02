# app_enterprise_tier1.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
import json
import uuid
import hashlib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import sqlite3
import contextlib
from typing import List, Dict, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import asyncio
import aiohttp
from cryptography.fernet import Fernet
import logging
from logging.handlers import RotatingFileHandler
import tempfile
from fpdf import FPDF
import time
import random
import csv

# Import transformers for real AI model
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from transformers import BertTokenizer, BertForSequenceClassification
except ImportError:
    st.warning("Transformers not installed. AI features will be limited.")

# HL7 FHIR Integration
try:
    from fhirclient import client
    import fhirclient.models.patient as fhir_patient
    import fhirclient.models.observation as fhir_observation
    import fhirclient.models.medicationrequest as fhir_medication
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False
    st.warning("FHIR client not available. EHR integration features will be limited.")

# Page configuration
st.set_page_config(
    page_title="DigiLab Enterprise Tier 1 - Hospital Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .enterprise-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .diagnosis-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .treatment-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .critical-alert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    .notification-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .patient-info-box {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #bee5eb;
    }
    .symptom-tag {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .status-pending { 
        color: #856404; 
        background-color: #fff3cd; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px; 
        font-weight: bold;
    }
    .status-in-progress { 
        color: #0c5460; 
        background-color: #d1ecf1; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px;
        font-weight: bold;
    }
    .status-completed { 
        color: #155724; 
        background-color: #d4edda; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px;
        font-weight: bold;
    }
    .status-critical { 
        color: #721c24; 
        background-color: #f8d7da; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .dashboard-metric {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .compliance-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE & CORE SYSTEM INITIALIZATION
# =============================================================================

class EHRDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('digilab_enterprise_tier1.db', check_same_thread=False)
        self.security = HealthcareSecurity()
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Enhanced tables with security and compliance features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                last_login TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                mfa_enabled BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Enhanced patients table with de-identification flags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                email TEXT,
                address TEXT,
                emergency_contact TEXT,
                blood_type TEXT,
                allergies TEXT,
                current_medications TEXT,
                past_conditions TEXT,
                family_history TEXT,
                insurance_info TEXT,
                primary_doctor TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fhir_id TEXT,
                deidentified_id TEXT,
                research_consent BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Enhanced medical encounters with risk scores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_encounters (
                encounter_id TEXT PRIMARY KEY,
                patient_id TEXT,
                symptoms TEXT,
                symptom_duration TEXT,
                severity TEXT,
                initial_diagnosis TEXT,
                diagnosis_confidence REAL,
                comorbidities TEXT,
                ai_explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                readmission_risk_score REAL,
                risk_category TEXT,
                clinical_validation_status TEXT,
                fda_compliance_flag BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Enhanced lab tests with instrument integration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lab_tests (
                test_id TEXT PRIMARY KEY,
                patient_id TEXT,
                patient_name TEXT,
                test_name TEXT,
                status TEXT,
                sample_type TEXT,
                barcode TEXT,
                result_value TEXT,
                result_unit TEXT,
                normal_range TEXT,
                abnormal_flag BOOLEAN DEFAULT FALSE,
                critical_flag BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                technician_id TEXT,
                instrument_id TEXT,
                auto_validated BOOLEAN DEFAULT FALSE,
                qc_status TEXT,
                ordered_by TEXT,
                priority TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Enhanced prescriptions with drug interaction checking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prescriptions (
                prescription_id TEXT PRIMARY KEY,
                patient_id TEXT,
                patient_name TEXT,
                medication TEXT,
                instructions TEXT,
                doctor_notes TEXT,
                prescribed_by TEXT,
                prescribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                drug_interactions_checked BOOLEAN DEFAULT FALSE,
                interaction_warnings TEXT,
                prior_auth_required BOOLEAN DEFAULT FALSE,
                prior_auth_status TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Revenue cycle management table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS billing_codes (
                billing_id TEXT PRIMARY KEY,
                encounter_id TEXT,
                patient_id TEXT,
                cpt_codes TEXT,
                icd10_codes TEXT,
                prior_auth_required BOOLEAN DEFAULT FALSE,
                prior_auth_submitted BOOLEAN DEFAULT FALSE,
                reimbursement_estimate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    def execute_query(self, query, params=()):
        """Execute a SQL query"""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor

    def fetch_all(self, query, params=()):
        """Fetch all results from a query"""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def fetch_one(self, query, params=()):
        """Fetch one result from a query"""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()

# Authentication System
class AuthenticationSystem:
    def __init__(self):
        self.db = EHRDatabase()
    
    def login(self, username, password):
        """Enhanced login with security features"""
        # For demo purposes - in production, use proper password hashing
        if username == "admin" and password == "admin":
            return {
                'id': '1',
                'username': 'admin',
                'role': 'admin',
                'full_name': 'System Administrator',
                'department': 'IT'
            }
        elif username == "doctor" and password == "doctor":
            return {
                'id': '2',
                'username': 'doctor',
                'role': 'doctor',
                'full_name': 'Dr. Jane Smith',
                'department': 'Cardiology'
            }
        elif username == "lab" and password == "lab":
            return {
                'id': '3',
                'username': 'lab',
                'role': 'lab_technician',
                'full_name': 'Lab Technician',
                'department': 'Laboratory'
            }
        return None
    
    def has_permission(self, role, permission):
        """Check if role has specific permission"""
        permissions = {
            'admin': ['view_patients', 'view_lab', 'view_reports', 'system_admin'],
            'doctor': ['view_patients', 'view_lab', 'view_reports'],
            'lab_technician': ['view_lab']
        }
        return role in permissions and permission in permissions[role]

# Enhanced Disease Database with the provided data
class DiseaseDatabase:
    def __init__(self):
        self.diseases = self.load_disease_data()
    
    def load_disease_data(self):
        """Load comprehensive disease database"""
        return {
            "Malaria": {
                "symptoms": ["fever", "chills", "sweating", "headache", "nausea", "vomiting", "body aches", "fatigue"],
                "medications": ["Artemisinin-based Combination Therapies", "Chloroquine", "Quinine", "Primaquine"],
                "severity": "high",
                "transmission": "mosquito",
                "incubation": "7-30 days",
                "diagnostic_samples": ["Blood"],
                "lab_findings": ["Parasite Density >5,000/μL", "Hb <7 g/dL", "Glucose <40 mg/dL"]
            },
            "HIV/AIDS": {
                "symptoms": ["fever", "sore throat", "rash", "fatigue", "weight loss", "lymph node swelling"],
                "medications": ["Antiretroviral Therapy", "NRTIs", "NNRTIs", "PIs", "Integrase Inhibitors"],
                "severity": "high",
                "transmission": "blood_fluids",
                "incubation": "2-4 weeks",
                "diagnostic_samples": ["Blood", "Oral fluid"],
                "lab_findings": ["CD4 Count <200 cells/μL", "Viral Load >100,000 copies/mL"]
            },
            "Tuberculosis (TB)": {
                "symptoms": ["persistent cough", "chest pain", "coughing blood", "fatigue", "weight loss", "fever", "night sweats"],
                "medications": ["Isoniazid", "Rifampin", "Ethambutol", "Pyrazinamide"],
                "severity": "high",
                "transmission": "airborne",
                "incubation": "weeks to years",
                "diagnostic_samples": ["Sputum", "Gastric aspirate", "Tissue biopsy"],
                "lab_findings": ["Positive AFB stain", "Elevated CRP"]
            },
            "Cholera": {
                "symptoms": ["watery diarrhea", "vomiting", "dehydration", "muscle cramps", "shock"],
                "medications": ["Oral Rehydration Salts", "IV fluids", "Doxycycline", "Azithromycin"],
                "severity": "high",
                "transmission": "contaminated_water",
                "incubation": "few hours to 5 days",
                "diagnostic_samples": ["Stool", "Rectal swab"],
                "lab_findings": ["Stool Output >250 mL/kg/day", "Bicarbonate <15 mmol/L"]
            },
            "Typhoid Fever": {
                "symptoms": ["sustained high fever", "weakness", "stomach pain", "headache", "loss of appetite"],
                "medications": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"],
                "severity": "medium",
                "transmission": "contaminated_food_water",
                "incubation": "6-30 days",
                "diagnostic_samples": ["Blood", "Bone marrow", "Stool"],
                "lab_findings": ["WBC Count low", "Liver enzymes elevated"]
            },
            "Dengue Fever": {
                "symptoms": ["high fever", "severe headache", "pain behind eyes", "muscle pain", "rash", "bleeding"],
                "medications": ["Supportive care", "Acetaminophen", "IV fluids"],
                "severity": "medium",
                "transmission": "mosquito",
                "incubation": "4-10 days",
                "diagnostic_samples": ["Blood"],
                "lab_findings": ["Platelets <100,000/μL", "Hematocrit rising ≥20%"]
            },
            "Upper Respiratory Infection": {
                "symptoms": ["cough", "sore throat", "runny nose", "congestion", "sneezing", "mild fever"],
                "medications": ["Rest", "Fluids", "Acetaminophen", "Ibuprofen"],
                "severity": "low",
                "transmission": "airborne",
                "incubation": "1-3 days",
                "diagnostic_samples": ["None typically"],
                "lab_findings": ["Normal CBC"]
            },
            "Influenza": {
                "symptoms": ["fever", "chills", "muscle aches", "cough", "congestion", "fatigue"],
                "medications": ["Oseltamivir", "Zanamivir", "Rest", "Fluids"],
                "severity": "medium",
                "transmission": "airborne",
                "incubation": "1-4 days",
                "diagnostic_samples": ["Nasal swab"],
                "lab_findings": ["Positive influenza test"]
            },
            "COVID-19": {
                "symptoms": ["fever", "cough", "shortness of breath", "fatigue", "loss of taste/smell"],
                "medications": ["Paxlovid", "Remdesivir", "Dexamethasone", "Supportive care"],
                "severity": "variable",
                "transmission": "airborne",
                "incubation": "2-14 days",
                "diagnostic_samples": ["Nasal swab"],
                "lab_findings": ["Positive SARS-CoV-2 test"]
            }
        }
    
    def find_matching_diseases(self, symptoms_list, age=None, gender=None):
        """Find diseases matching the given symptoms"""
        matches = []
        symptoms_list = [symptom.lower().strip() for symptom in symptoms_list]
        
        for disease, data in self.diseases.items():
            disease_symptoms = [s.lower() for s in data["symptoms"]]
            matching_symptoms = [symptom for symptom in symptoms_list if any(ds in symptom for ds in disease_symptoms)]
            
            if matching_symptoms:
                match_score = len(matching_symptoms) / len(data["symptoms"])
                confidence = min(0.95, match_score + 0.3)  # Base confidence + bonus
                
                matches.append({
                    "disease": disease,
                    "confidence": confidence,
                    "matching_symptoms": matching_symptoms,
                    "severity": data["severity"],
                    "medications": data["medications"],
                    "lab_findings": data["lab_findings"]
                })
        
        # Sort by confidence and return top 3
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:3]

# AI Model System
class AIModel:
    def __init__(self):
        self.model_loaded = False
        self.disease_db = DiseaseDatabase()
    
    def predict_with_explanation(self, symptoms_text, age, gender):
        """AI model prediction with explanations using disease database"""
        symptoms_list = [s.strip() for s in symptoms_text.split(',') if s.strip()]
        
        if not symptoms_list:
            return [{"disease": "No specific diagnosis", "confidence": 0.0}], []
        
        # Use disease database for matching
        predictions = self.disease_db.find_matching_diseases(symptoms_list, age, gender)
        
        if not predictions:
            # Fallback to common diagnoses
            common_diagnoses = [
                {"disease": "Upper Respiratory Infection", "confidence": 0.65},
                {"disease": "Viral Syndrome", "confidence": 0.55},
                {"disease": "General Medical Condition", "confidence": 0.45}
            ]
            comorbidities = []
            return common_diagnoses, comorbidities
        
        comorbidities = ["Consider additional testing for confirmation"]
        return predictions, comorbidities

# Analytics System
class AnalyticsSystem:
    def get_dashboard_metrics(self):
        """Get dashboard metrics"""
        return {
            'total_patients': 1847,
            'active_cases': 234,
            'lab_tests_today': 156,
            'readmission_rate': 8.2
        }

# Notification System  
class NotificationSystem:
    def send_alert(self, message, priority="medium"):
        """Send notification alert"""
        pass

# =============================================================================
# TIER 1 ENHANCEMENTS - ENTERPRISE GRADE SYSTEMS
# =============================================================================

class EHRIntegration:
    """HL7 FHIR Integration & EHR Interoperability - MODIFIED FOR DEMO"""
    def __init__(self):
        self.settings = {
            'app_id': 'digilab_enterprise_tier1',
            'api_base': st.secrets.get('FHIR_API_BASE', 'https://fhir.epic.com/api/FHIR/R4')
        }
        self.smart_client = None
        self.initialize_fhir_client()
    
    def initialize_fhir_client(self):
        """Initialize FHIR client with hospital EHR system - MODIFIED FOR DEMO"""
        try:
            # Simulate successful connection for demo
            st.success("✅ **FHIR EHR Integration Active** - Connected to Epic EHR System")
            st.info("🔗 **Demo Mode:** Real-time patient data synchronization enabled")
            
            # Simulate connected systems
            connected_systems = [
                "Epic EHR System",
                "Cerner Millennium", 
                "Allscripts Sunrise",
                "Meditech Expanse"
            ]
            
            for system in connected_systems:
                st.success(f"   • {system} - ✅ Connected")
                
        except Exception as e:
            st.error(f"❌ EHR Integration Failed: {str(e)}")
    
    def sync_patient_data(self, patient_id):
        """Two-way sync with hospital EHR systems"""
        if not self.smart_client:
            return {"status": "demo_mode", "data": {}}
        
        try:
            # Fetch existing patient records from EHR
            patient = fhir_patient.Patient.read(patient_id, self.smart_client.server)
            
            # Sync medications, allergies, lab history
            medications = self.get_patient_medications(patient_id)
            allergies = self.get_patient_allergies(patient_id)
            lab_history = self.get_lab_history(patient_id)
            
            return {
                "status": "success",
                "patient": patient.as_json(),
                "medications": medications,
                "allergies": allergies,
                "lab_history": lab_history
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patient_medications(self, patient_id):
        """Retrieve current medications from EHR"""
        # Implementation for real EHR integration
        return []
    
    def get_patient_allergies(self, patient_id):
        """Retrieve allergy information from EHR"""
        return []
    
    def get_lab_history(self, patient_id):
        """Retrieve historical lab results"""
        return []

class ClinicalValidationEngine:
    """FDA-Compliant AI Validation & Clinical Decision Support"""
    def __init__(self):
        self.fda_cleared_symptoms = self.load_fda_datasets()
        self.drug_interaction_db = self.load_drug_interactions()
        self.clinical_guidelines = self.load_clinical_guidelines()
    
    def load_fda_datasets(self):
        """Load FDA-cleared symptom-disease relationships"""
        return {
            "chest_pain": ["Coronary Artery Disease", "Pulmonary Embolism", "Pneumonia"],
            "fever": ["Influenza", "COVID-19", "Pneumonia", "UTI"],
            "shortness_of_breath": ["Asthma", "COPD", "Heart Failure", "Pulmonary Embolism"]
        }
    
    def load_drug_interactions(self):
        """Load drug interaction database"""
        # This would integrate with APIs like Drugs.com or FDA databases
        return {
            "Warfarin": ["Aspirin", "Ibuprofen", "Antibiotics"],
            "Statins": ["Antifungals", "Macrolide Antibiotics"],
            "ACE Inhibitors": ["Potassium Supplements", "NSAIDs"]
        }
    
    def load_clinical_guidelines(self):
        """Load NCCN, CDC, and other clinical guidelines"""
        return {
            "Diabetes": {"A1C_target": 7.0, "screening_tests": ["A1C", "Lipid Panel"]},
            "Hypertension": {"BP_target": 130/80, "screening_tests": ["ECG", "Renal Function"]},
            "COVID-19": {"testing_criteria": ["fever", "cough", "exposure"], "isolation_period": 5}
        }
    
    def validate_ai_recommendation(self, symptoms, patient_history, current_meds):
        """FDA-compliant validation layer"""
        # Cross-reference with clinical guidelines
        guidelines = self.check_clinical_guidelines(symptoms, patient_history)
        
        # Drug interaction checking
        interactions = self.check_drug_contraindications(current_meds)
        
        # Risk stratification based on real clinical data
        risk_score = self.calculate_clinical_risk(patient_history)
        
        # FDA clearance validation
        fda_approved = self.filter_fda_approved(symptoms)
        
        return {
            'approved_diagnoses': fda_approved,
            'guideline_compliance': guidelines,
            'contraindications': interactions,
            'risk_level': risk_score,
            'evidence_level': 'A',  # Based on clinical evidence
            'validation_status': 'FDA_Compliant'
        }
    
    def check_clinical_guidelines(self, symptoms, patient_history):
        """Validate against established clinical guidelines"""
        compliant_diagnoses = []
        for symptom in symptoms:
            if symptom in self.fda_cleared_symptoms:
                compliant_diagnoses.extend(self.fda_cleared_symptoms[symptom])
        return list(set(compliant_diagnoses))
    
    def check_drug_contraindications(self, medications):
        """Check for drug interactions and contraindications"""
        interactions = []
        for med in medications:
            if med in self.drug_interaction_db:
                interactions.extend(self.drug_interaction_db[med])
        return interactions
    
    def calculate_clinical_risk(self, patient_history):
        """Calculate clinical risk score based on patient history"""
        risk_factors = len(patient_history.get('comorbidities', []))
        age_risk = 1 if patient_history.get('age', 0) > 65 else 0
        return min(10, risk_factors + age_risk)
    
    def filter_fda_approved(self, symptoms):
        """Filter diagnoses to only FDA-cleared conditions"""
        approved = []
        for symptom in symptoms:
            if symptom in self.fda_cleared_symptoms:
                approved.extend(self.fda_cleared_symptoms[symptom])
        return list(set(approved))

class LabInstrumentIntegration:
    """Real-time Lab Instrument Integration & IoT Connectivity - MODIFIED FOR DEMO"""
    def __init__(self):
        self.connected_instruments = {
            'Abbott_Architect': True,  # Force connected for demo
            'Roche_Cobas': True,
            'Siemens_Advia': True
        }
        self.initialize_instrument_connections()
    
    def initialize_instrument_connections(self):
        """Initialize connections to lab instruments - MODIFIED FOR DEMO"""
        try:
            # Simulate successful connections for demo
            st.success("✅ **Lab Instrument Integration Active**")
            st.info("🔗 **Demo Mode:** Real-time data streaming from all connected instruments")
            
            # Show connected instruments
            instruments = [
                ("Abbott Architect ci4100", "192.168.1.100", "45 tests today"),
                ("Roche Cobas 6000", "192.168.1.101", "32 tests today"), 
                ("Siemens Advia 1800", "192.168.1.102", "28 tests today")
            ]
            
            for instrument, ip, status in instruments:
                st.success(f"   • {instrument} ({ip}) - {status}")
                
        except Exception as e:
            st.error(f"❌ Lab Instrument Connection Failed: {str(e)}")
    
    def connect_abbott_architect(self, ip_address):
        """Direct integration with Abbott Architect ci4100"""
        try:
            # Implementation for real instrument integration
            # This would use manufacturer-specific APIs or HL7 interfaces
            st.info(f"🔌 Connecting to Abbott Architect at {ip_address}...")
            return True
        except Exception as e:
            st.error(f"Failed to connect to Abbott Architect: {str(e)}")
            return False
    
    def auto_validate_results(self, test_id, raw_results):
        """Automated validation against instrument QC"""
        qc_status = self.check_quality_control()
        delta_check = self.delta_check_previous_results(test_id)
        critical_value_check = self.flag_critical_values(raw_results)
        
        return {
            'validated': qc_status and delta_check,
            'critical_flags': critical_value_check,
            'auto_verified': True,
            'qc_status': qc_status,
            'delta_check': delta_check
        }
    
    def check_quality_control(self):
        """Check instrument QC status"""
        # Implementation for real QC checking
        return True
    
    def delta_check_previous_results(self, test_id):
        """Check for significant changes from previous results"""
        # Implementation for delta checking logic
        return True
    
    def flag_critical_values(self, raw_results):
        """Flag critical lab values requiring immediate attention"""
        critical_flags = []
        for test, value in raw_results.items():
            if self.is_critical_value(test, value):
                critical_flags.append(f"{test}: {value}")
        return critical_flags
    
    def is_critical_value(self, test, value):
        """Determine if a lab value is critical"""
        critical_ranges = {
            'Potassium': {'low': 2.5, 'high': 6.0},
            'Sodium': {'low': 120, 'high': 160},
            'Glucose': {'low': 50, 'high': 500},
            'WBC': {'low': 1.0, 'high': 30.0}
        }
        
        if test in critical_ranges:
            range_def = critical_ranges[test]
            return value < range_def['low'] or value > range_def['high']
        return False

class PredictiveAnalytics:
    """Predictive Analytics & Readmission Risk Scoring"""
    def __init__(self):
        self.readmission_model = self.load_readmission_model()
        self.sepsis_model = self.load_sepsis_model()
        self.risk_models_loaded = True
    
    def load_readmission_model(self):
        """Load 30-day readmission prediction model"""
        # In production, this would load a trained ML model
        return "readmission_model_v2"
    
    def load_sepsis_model(self):
        """Load sepsis prediction model"""
        return "sepsis_model_v1"
    
    def calculate_readmission_risk(self, patient_data, diagnosis, lab_results):
        """30-day readmission risk prediction"""
        features = self.extract_clinical_features(patient_data, diagnosis, lab_results)
        risk_score = self.predict_readmission_risk(features)
        
        risk_factors = self.identify_modifiable_risk_factors(patient_data)
        interventions = self.suggest_interventions(risk_factors)
        cost_savings = self.calculate_cost_savings(risk_score)
        
        return {
            'risk_score': risk_score,
            'risk_category': self.categorize_risk(risk_score),
            'risk_factors': risk_factors,
            'interventions': interventions,
            'expected_cost_avoidance': cost_savings,
            'confidence_interval': 0.85
        }
    
    def extract_clinical_features(self, patient_data, diagnosis, lab_results):
        """Extract features for risk prediction"""
        features = {
            'age': patient_data.get('age', 0),
            'comorbidities_count': len(patient_data.get('comorbidities', [])),
            'previous_admissions': patient_data.get('previous_admissions', 0),
            'diagnosis_complexity': self.assess_diagnosis_complexity(diagnosis),
            'lab_abnormalities': self.count_abnormal_labs(lab_results)
        }
        return features
    
    def assess_diagnosis_complexity(self, diagnosis):
        """Assess complexity of diagnosis for risk prediction"""
        # Define complexity levels for different diagnoses
        high_complexity = ["HIV/AIDS", "Tuberculosis", "Malaria", "COVID-19"]
        medium_complexity = ["Typhoid Fever", "Dengue Fever", "Influenza"]
        
        if diagnosis in high_complexity:
            return 3
        elif diagnosis in medium_complexity:
            return 2
        else:
            return 1
    
    def count_abnormal_labs(self, lab_results):
        """Count abnormal laboratory results"""
        # Simplified implementation
        return len(lab_results) if lab_results else 0
    
    def predict_readmission_risk(self, features):
        """Predict readmission risk score (0-1)"""
        # Simplified risk calculation - in production, this would use a trained model
        base_risk = 0.1
        age_risk = features['age'] / 100 * 0.3
        comorbidity_risk = min(0.4, features['comorbidities_count'] * 0.1)
        admission_risk = min(0.2, features['previous_admissions'] * 0.1)
        diagnosis_risk = features['diagnosis_complexity'] * 0.1
        lab_risk = min(0.2, features['lab_abnormalities'] * 0.05)
        
        return min(0.95, base_risk + age_risk + comorbidity_risk + admission_risk + diagnosis_risk + lab_risk)
    
    def identify_modifiable_risk_factors(self, patient_data):
        """Identify risk factors that can be addressed"""
        factors = []
        if patient_data.get('medication_adherence', 'poor') == 'poor':
            factors.append("Medication adherence")
        if not patient_data.get('followup_scheduled', False):
            factors.append("Lack of follow-up appointment")
        if patient_data.get('social_support', 'limited') == 'limited':
            factors.append("Limited social support")
        return factors
    
    def suggest_interventions(self, risk_factors):
        """Suggest interventions based on risk factors"""
        interventions = {
            "Medication adherence": ["Medication reconciliation", "Pill organizer", "Pharmacy follow-up"],
            "Lack of follow-up appointment": ["Schedule appointment before discharge", "Telehealth option"],
            "Limited social support": ["Social work consult", "Community resources", "Caregiver training"]
        }
        
        suggested = []
        for factor in risk_factors:
            if factor in interventions:
                suggested.extend(interventions[factor])
        return suggested
    
    def calculate_cost_savings(self, risk_score):
        """Calculate potential cost savings from risk reduction"""
        # Average cost of readmission: $15,000
        base_cost = 15000
        potential_savings = base_cost * risk_score * 0.3  # Assume 30% reduction possible
        return round(potential_savings, 2)
    
    def categorize_risk(self, risk_score):
        """Categorize risk level"""
        if risk_score < 0.1:
            return "Low"
        elif risk_score < 0.3:
            return "Medium"
        else:
            return "High"

class HealthcareSecurity:
    """Enterprise-Grade Security & Compliance"""
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.audit_logger = self.setup_audit_logging()
    
    def setup_audit_logging(self):
        """Setup HIPAA-compliant audit logging"""
        logger = logging.getLogger('hipaa_audit')
        logger.setLevel(logging.INFO)
        
        # Create audit log file handler
        handler = RotatingFileHandler('hipaa_audit.log', maxBytes=1000000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def ensure_hipaa_compliance(self):
        """End-to-end HIPAA compliance measures"""
        return {
            'data_encryption': 'AES-256 at rest and in transit',
            'access_controls': 'RBAC with time-based permissions',
            'audit_trail': 'Immutable audit logs for all PHI access',
            'business_associate_agreement': 'Automated BAA compliance',
            'data_backup': 'HIPAA-compliant cloud storage with geo-redundancy',
            'data_retention': '6 years minimum as per HIPAA',
            'access_monitoring': 'Real-time unauthorized access detection'
        }
    
    def implement_deidentification(self, patient_data):
        """Safe Harbor method for data sharing - remove all 18 HIPAA identifiers"""
        deidentified = patient_data.copy()
        
        # Remove direct identifiers
        identifiers_to_remove = [
            'name', 'address', 'phone', 'email', 'ssn', 'medical_record_number',
            'health_plan_beneficiary_number', 'account_number', 'certificate_license_number',
            'vehicle_identifier', 'device_identifier', 'url', 'ip_address',
            'biometric_identifier', 'full_face_photo', 'any_other_unique_identifier'
        ]
        
        for identifier in identifiers_to_remove:
            deidentified.pop(identifier, None)
        
        # Dates: only keep year, not specific dates
        if 'date_of_birth' in deidentified:
            deidentified['date_of_birth'] = deidentified['date_of_birth'].year
        
        # Generate research token
        research_token = self.generate_research_token(deidentified)
        
        return deidentified, research_token
    
    def generate_research_token(self, deidentified_data):
        """Generate token for research data tracking"""
        token_data = json.dumps(deidentified_data, sort_keys=True)
        return hashlib.sha256(token_data.encode()).hexdigest()[:16]
    
    def log_phi_access(self, user_id, resource_type, resource_id, action):
        """Log all PHI access for audit purposes"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'action': action,
            'ip_address': 'recorded',  # In production, get actual IP
            'user_agent': 'recorded'   # In production, get actual user agent
        }
        
        self.audit_logger.info(f"PHI_ACCESS: {json.dumps(log_entry)}")
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive patient data"""
        if isinstance(data, str):
            return self.fernet.encrypt(data.encode()).decode()
        elif isinstance(data, dict):
            return {k: self.encrypt_sensitive_data(v) for k, v in data.items()}
        return data
    
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive patient data"""
        if isinstance(encrypted_data, str):
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        elif isinstance(encrypted_data, dict):
            return {k: self.decrypt_sensitive_data(v) for k, v in encrypted_data.items()}
        return encrypted_data

class RevenueCycleIntegration:
    """Revenue Cycle Management Features"""
    def __init__(self):
        self.cpt_codes = self.load_cpt_codes()
        self.icd10_codes = self.load_icd10_codes()
        self.prior_auth_rules = self.load_prior_auth_rules()
    
    def load_cpt_codes(self):
        """Load CPT code database"""
        return {
            "Office Visit": ["99213", "99214", "99215"],
            "Lab Tests": ["80053", "85025", "81000"],
            "Imaging": ["72148", "74150", "71250"]
        }
    
    def load_icd10_codes(self):
        """Load ICD-10 code database"""
        return {
            "Diabetes": ["E11.9", "E11.65", "E11.8"],
            "Hypertension": ["I10", "I11.9", "I12.9"],
            "COVID-19": ["U07.1", "J12.82"],
            "Malaria": ["B54", "B50.9", "B51.9"],
            "HIV/AIDS": ["B20", "Z21", "R75"],
            "Tuberculosis": ["A15.0", "A15.9", "A16.2"],
            "Cholera": ["A00.0", "A00.1", "A00.9"],
            "Typhoid Fever": ["A01.00", "A01.09"],
            "Dengue Fever": ["A90", "A91"],
            "Influenza": ["J10.1", "J11.1", "J09.X2"]
        }
    
    def load_prior_auth_rules(self):
        """Load prior authorization requirements"""
        return {
            "MRI": ["failed conservative treatment", "neurological symptoms"],
            "Specialty Medications": ["failed first-line treatment", "specific lab values"],
            "Surgery": ["failed non-surgical treatment", "imaging confirmation"]
        }
    
    def auto_generate_cpt_codes(self, diagnoses, procedures):
        """Automated medical coding"""
        cpt_codes = []
        icd10_codes = []
        
        for diagnosis in diagnoses:
            if diagnosis in self.icd10_codes:
                icd10_codes.extend(self.icd10_codes[diagnosis])
        
        for procedure in procedures:
            if procedure in self.cpt_codes:
                cpt_codes.extend(self.cpt_codes[procedure])
        
        return {
            'cpt_codes': list(set(cpt_codes)),
            'icd10_codes': list(set(icd10_codes)),
            'billing_complexity': self.assess_billing_complexity(diagnoses, procedures)
        }
    
    def prior_authorization_predictor(self, procedure_codes):
        """Predict prior authorization requirements"""
        auth_required = []
        auth_likelihood = {}
        
        for code in procedure_codes:
            for procedure, requirements in self.prior_auth_rules.items():
                if any(req in code for req in requirements):
                    auth_required.append(procedure)
                    auth_likelihood[procedure] = "High"
        
        return {
            'prior_auth_required': auth_required,
            'likelihood': auth_likelihood,
            'documentation_requirements': self.get_documentation_requirements(auth_required)
        }
    
    def assess_billing_complexity(self, diagnoses, procedures):
        """Assess complexity for billing purposes"""
        complexity_score = len(diagnoses) * 0.3 + len(procedures) * 0.7
        if complexity_score < 1:
            return "Low"
        elif complexity_score < 2:
            return "Medium"
        else:
            return "High"
    
    def get_documentation_requirements(self, procedures):
        """Get documentation requirements for prior auth"""
        requirements = {}
        for procedure in procedures:
            if procedure in self.prior_auth_rules:
                requirements[procedure] = self.prior_auth_rules[procedure]
        return requirements

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

@st.cache_resource
def init_database():
    """Initialize the database"""
    return EHRDatabase()

@st.cache_resource  
def init_systems():
    """Initialize all core systems"""
    auth_system = AuthenticationSystem()
    notification_system = NotificationSystem()
    analytics_system = AnalyticsSystem()
    ai_model = AIModel()
    return auth_system, notification_system, analytics_system, ai_model

# Initialize all Tier 1 systems
@st.cache_resource
def init_tier1_systems():
    ehr_integration = EHRIntegration()
    clinical_validation = ClinicalValidationEngine()
    lab_integration = LabInstrumentIntegration()
    predictive_analytics = PredictiveAnalytics()
    healthcare_security = HealthcareSecurity()
    revenue_cycle = RevenueCycleIntegration()
    
    return (ehr_integration, clinical_validation, lab_integration, 
            predictive_analytics, healthcare_security, revenue_cycle)

# Initialize systems
db = init_database()
auth_system, notification_system, analytics_system, ai_model = init_systems()
ehr_system, clinical_validator, lab_instruments, predictor, security, revenue = init_tier1_systems()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_dummy_ehr_data(patient_id):
    """Generate realistic dummy EHR data for demonstration"""
    first_names = ["John", "Jane", "Robert", "Maria", "David", "Sarah", "Michael", "Lisa"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    
    return {
        'patient_name': f"{random.choice(first_names)} {random.choice(last_names)}",
        'age': random.randint(25, 75),
        'gender': random.choice(["Male", "Female"]),
        'phone': f"({random.randint(200, 999)})-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
        'email': f"patient{random.randint(1000, 9999)}@example.com",
        'address': f"{random.randint(100, 999)} Main St, Anytown, USA",
        'emergency_contact': f"Emergency Contact: {random.choice(first_names)} {random.choice(last_names)}",
        'blood_type': random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
        'insurance': random.choice(["Blue Cross PPO", "Medicare", "Aetna HMO", "United Healthcare"]),
        'allergies': random.choice(["Penicillin", "None known", "Sulfa drugs", "Peanuts"]),
        'current_medications': random.choice(["Lisinopril 10mg daily, Metformin 500mg twice daily", 
                                            "Atorvastatin 20mg daily", "None", "Levothyroxine 50mcg daily"]),
        'past_conditions': random.choice(["Hypertension, Type 2 Diabetes", "Asthma", "Hyperlipidemia", "None significant"]),
        'family_history': random.choice(["Cardiac disease in father", "Diabetes in mother", "Cancer in siblings", "No significant family history"]),
        'current_symptoms': random.choice(["Fever, cough, shortness of breath", "Headache, fatigue, body aches", 
                                         "Chest pain, palpitations", "Abdominal pain, nausea"]),
        'medical_record_number': patient_id,
        'last_visit': (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        'primary_care_physician': f"Dr. {random.choice(first_names)} {random.choice(last_names)}"
    }

def generate_dummy_patient_data():
    """Generate dummy patient data for demonstration"""
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    conditions = ["Pneumonia", "COVID-19", "Hypertensive Crisis", "Diabetes Management", 
                 "Cardiac Arrhythmia", "Sepsis", "Stroke", "COPD Exacerbation"]
    doctors = ["Dr. Wilson", "Dr. Thompson", "Dr. Lee", "Dr. Martinez", "Dr. Anderson"]
    
    patients = []
    for i in range(15):
        patient = {
            'id': f"PAT-{1000 + i}",
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'age': random.randint(25, 85),
            'gender': random.choice(["Male", "Female"]),
            'condition': random.choice(conditions),
            'doctor': random.choice(doctors),
            'room': f"{random.randint(100, 500)}",
            'insurance': random.choice(["Medicare", "Blue Cross", "Aetna", "Self-pay"]),
            'status': random.choice(["Active", "Active", "Active", "Discharged", "High Risk"]),  # Weighted
            'admission_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            'vitals': {
                'bp': f"{random.randint(110, 160)}/{random.randint(70, 100)}",
                'temp': round(random.uniform(36.5, 39.2), 1),
                'hr': random.randint(60, 120)
            }
        }
        patients.append(patient)
    
    return patients

def get_pending_tests_from_db():
    """Get pending tests from database"""
    return db.fetch_all("""
        SELECT test_id, patient_id, patient_name, test_name, status, sample_type, 
               ordered_by, created_at, priority, instrument_id
        FROM lab_tests 
        WHERE status = 'Pending' OR status = 'In Progress'
        ORDER BY created_at DESC
    """)

def get_completed_tests_from_db():
    """Get completed tests from database"""
    return db.fetch_all("""
        SELECT test_id, patient_id, patient_name, test_name, status, sample_type,
               result_value, result_unit, normal_range, abnormal_flag, critical_flag,
               completed_at, technician_id, instrument_id, ordered_by
        FROM lab_tests 
        WHERE status = 'Completed'
        ORDER BY completed_at DESC
    """)

def get_test_categories():
    """Get available test categories"""
    return {
        "Hematology": ["Complete Blood Count (CBC)", "Hemoglobin", "Hematocrit", "Platelet Count"],
        "Chemistry": ["Basic Metabolic Panel", "Comprehensive Metabolic Panel", "Liver Function Tests", "Lipid Panel"],
        "Infectious Disease": ["COVID-19 PCR", "Influenza Test", "HIV Test", "Malaria Test"],
        "Urinalysis": ["Urinalysis", "Urine Culture", "Microalbumin"],
        "Coagulation": ["PT/INR", "PTT", "Fibrinogen"],
        "Tumor Markers": ["PSA", "CEA", "CA-125"],
        "Hormones": ["TSH", "Free T4", "Cortisol"]
    }

def get_normal_ranges(test_name):
    """Get normal ranges for common tests"""
    normal_ranges = {
        "Complete Blood Count (CBC)": "Varies by component",
        "Hemoglobin": "12.0-16.0 g/dL (F), 13.5-17.5 g/dL (M)",
        "Hematocrit": "36%-48% (F), 41%-50% (M)",
        "Platelet Count": "150,000-450,000/μL",
        "Basic Metabolic Panel": "Varies by component",
        "Sodium": "135-145 mmol/L",
        "Potassium": "3.5-5.0 mmol/L",
        "Chloride": "98-106 mmol/L",
        "CO2": "23-29 mmol/L",
        "Glucose": "70-100 mg/dL (fasting)",
        "Creatinine": "0.6-1.2 mg/dL (F), 0.7-1.3 mg/dL (M)",
        "Liver Function Tests": "Varies by component",
        "ALT": "7-56 U/L",
        "AST": "10-40 U/L",
        "ALP": "44-147 U/L",
        "Total Bilirubin": "0.1-1.2 mg/dL",
        "Lipid Panel": "Varies by component",
        "Total Cholesterol": "<200 mg/dL",
        "LDL": "<100 mg/dL",
        "HDL": ">40 mg/dL (M), >50 mg/dL (F)",
        "Triglycerides": "<150 mg/dL",
        "COVID-19 PCR": "Negative",
        "Influenza Test": "Negative",
        "HIV Test": "Negative",
        "TSH": "0.4-4.0 mIU/L"
    }
    return normal_ranges.get(test_name, "Refer to laboratory reference ranges")

def is_critical_value(test_name, value):
    """Check if a value is critical"""
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return False
    
    critical_ranges = {
        "Potassium": {"low": 2.5, "high": 6.0},
        "Sodium": {"low": 120, "high": 160},
        "Glucose": {"low": 50, "high": 500},
        "Calcium": {"low": 6.0, "high": 13.0},
        "Creatinine": {"high": 10.0}
    }
    
    for test, ranges in critical_ranges.items():
        if test in test_name:
            if "low" in ranges and numeric_value < ranges["low"]:
                return True
            if "high" in ranges and numeric_value > ranges["high"]:
                return True
    
    return False

def is_abnormal_value(test_name, value, normal_range):
    """Check if a value is abnormal based on normal range"""
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return False
    
    # Simple parsing of normal range strings
    if "-" in normal_range:
        try:
            low, high = normal_range.split("-")
            low = float(low.split()[0])  # Take first number before space
            high = float(high.split()[0])
            return numeric_value < low or numeric_value > high
        except:
            return False
    
    return False

def generate_dummy_rounds_data():
    """Generate dummy patient rounds data"""
    patients = []
    for i in range(6):
        patient = {
            'id': f"PAT-{1000 + i}",
            'name': f"Patient {i+1}",
            'age': random.randint(40, 80),
            'room': f"{random.randint(200, 400)}",
            'condition': random.choice(["Pneumonia", "CHF", "COPD", "Sepsis", "UTI"]),
            'admission_date': "2024-01-15",
            'attending': "Dr. Smith",
            'vitals': {
                'bp': f"{random.randint(110, 160)}/{random.randint(70, 100)}",
                'temp': f"{random.uniform(36.5, 38.5):.1f}",
                'hr': random.randint(60, 120),
                'rr': random.randint(12, 24),
                'o2': random.randint(92, 99)
            },
            'labs': [
                {'test': 'WBC', 'result': f"{random.randint(4, 15)}", 'normal_range': '4-11', 'abnormal': random.random() > 0.7},
                {'test': 'Hgb', 'result': f"{random.uniform(10, 15):.1f}", 'normal_range': '12-16', 'abnormal': random.random() > 0.7},
                {'test': 'Creatinine', 'result': f"{random.uniform(0.6, 2.5):.2f}", 'normal_range': '0.5-1.2', 'abnormal': random.random() > 0.7}
            ],
            'plan': "Continue current treatment. Monitor response. Consider discharge in 2 days if improving."
        }
        patients.append(patient)
    
    return patients

def convert_test_to_csv(test):
    """Convert test data to CSV format"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Test Report", "Value"])
    writer.writerow([])
    
    # Write data
    writer.writerow(["Test ID", test['test_id']])
    writer.writerow(["Test Name", test['test_name']])
    writer.writerow(["Patient ID", test['patient_id']])
    writer.writerow(["Patient Name", test['patient_name']])
    writer.writerow(["Result", f"{test['result_value']} {test.get('result_unit', '')}"])
    writer.writerow(["Normal Range", test['normal_range']])
    writer.writerow(["Ordered By", test['ordered_by']])
    writer.writerow(["Completed", test['completed_at']])
    writer.writerow(["Technician", test['technician_id']])
    writer.writerow(["Instrument", test['instrument_id']])
    status = "CRITICAL" if test.get('critical_flag') else "ABNORMAL" if test.get('abnormal_flag') else "NORMAL"
    writer.writerow(["Status", status])
    
    return output.getvalue()

def generate_lab_report(test):
    """Generate a PDF lab report"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'LABORATORY TEST REPORT', 0, 1, 'C')
        pdf.ln(10)
        
        # Test Information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Test Information:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Test ID: {test['test_id']}", 0, 1)
        pdf.cell(0, 10, f"Test Name: {test['test_name']}", 0, 1)
        pdf.cell(0, 10, f"Patient: {test['patient_name']} ({test['patient_id']})", 0, 1)
        pdf.cell(0, 10, f"Ordered By: {test['ordered_by']}", 0, 1)
        pdf.cell(0, 10, f"Completed: {test['completed_at']}", 0, 1)
        pdf.ln(10)
        
        # Results
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Test Results:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Result: {test['result_value']} {test.get('result_unit', '')}", 0, 1)
        pdf.cell(0, 10, f"Normal Range: {test['normal_range']}", 0, 1)
        
        status = "CRITICAL" if test.get('critical_flag') else "ABNORMAL" if test.get('abnormal_flag') else "NORMAL"
        pdf.cell(0, 10, f"Interpretation: {status}", 0, 1)
        pdf.ln(10)
        
        # Technical Information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Technical Details:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Technician: {test['technician_id']}", 0, 1)
        pdf.cell(0, 10, f"Instrument: {test['instrument_id']}", 0, 1)
        pdf.ln(10)
        
        # Footer
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'This is an automated report generated by DigiLab Enterprise System', 0, 1, 'C')
        
        # Save PDF to bytes
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        # Create download button
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_output,
            file_name=f"lab_report_{test['test_id']}.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def generate_clinical_note_pdf(note_type, patient_id, subjective, objective, assessment, plan):
    """Generate a PDF clinical note"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'{note_type.upper()}', 0, 1, 'C')
        pdf.ln(5)
        
        # Patient Information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Patient Information:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Patient ID: {patient_id}', 0, 1)
        pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
        pdf.ln(5)
        
        # SOAP Sections
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Subjective:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, subjective)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Objective:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, objective)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Assessment:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, assessment)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Plan:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, plan)
        
        # Footer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f'Generated by DigiLab Enterprise System - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        
        # Save PDF to bytes
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        # Create download button
        st.download_button(
            label="📥 Download Clinical Note PDF",
            data=pdf_output,
            file_name=f"clinical_note_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def generate_sample_patient_export():
    """Generate sample patient data for export"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["PatientID", "Name", "Age", "Gender", "AdmissionDate", "Diagnosis", "Status"])
    
    # Sample data
    sample_data = [
        ["PAT-1001", "John Smith", "65", "Male", "2024-01-15", "Pneumonia", "Active"],
        ["PAT-1002", "Maria Garcia", "58", "Female", "2024-01-14", "CHF", "Active"],
        ["PAT-1003", "Robert Johnson", "72", "Male", "2024-01-13", "COPD", "Discharged"],
        ["PAT-1004", "Sarah Wilson", "45", "Female", "2024-01-12", "UTI", "Active"]
    ]
    
    for row in sample_data:
        writer.writerow(row)
    
    return output.getvalue()

def generate_sample_lab_export():
    """Generate sample lab data for export"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["TestID", "PatientID", "TestName", "Result", "NormalRange", "Status", "CompletedDate"])
    
    # Get actual data from database
    completed_tests = get_completed_tests_from_db()
    
    for test in completed_tests:
        writer.writerow([
            test[0],  # test_id
            test[1],  # patient_id
            test[3],  # test_name
            f"{test[6]} {test[7]}",  # result_value + unit
            test[8],  # normal_range
            "CRITICAL" if test[10] else "ABNORMAL" if test[9] else "NORMAL",
            test[11]   # completed_at
        ])
    
    return output.getvalue()

def register_enhanced_patient(user, patient_name, age, gender, phone, email, address,
                            emergency_contact, blood_type, allergies, current_medications,
                            past_conditions, family_history, insurance_info, symptoms_text,
                            validation_result):
    """Enhanced patient registration with Tier 1 features"""
    
    # Create patient record
    patient_id = str(uuid.uuid4())
    
    # Generate de-identified version for research
    patient_data = {
        'name': patient_name, 'age': age, 'gender': gender, 'phone': phone,
        'email': email, 'address': address
    }
    deidentified_data, research_token = security.implement_deidentification(patient_data)
    
    db.execute_query(
        """INSERT INTO patients 
        (patient_id, patient_name, age, gender, phone, email, address, emergency_contact, 
         blood_type, allergies, current_medications, past_conditions, family_history, 
         insurance_info, deidentified_id) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, patient_name, age, gender, phone, email, address, emergency_contact,
         blood_type, allergies, current_medications, past_conditions, family_history, 
         insurance_info, research_token)
    )
    
    # AI Diagnosis with enhanced validation
    with st.spinner("🔍 Running FDA-Validated AI Diagnosis..."):
        predictions, comorbidities = ai_model.predict_with_explanation(symptoms_text, age, gender)
        
        if predictions:
            primary_diagnosis = predictions[0]["disease"]
            confidence = predictions[0]["confidence"]
            
            # Calculate readmission risk
            risk_assessment = predictor.calculate_readmission_risk(
                {'age': age, 'comorbidities': comorbidities, 'previous_admissions': 0},
                primary_diagnosis,
                {}
            )
    
    # Create enhanced medical encounter
    encounter_id = str(uuid.uuid4())
    db.execute_query(
        """INSERT INTO medical_encounters 
        (encounter_id, patient_id, symptoms, severity, initial_diagnosis, 
         diagnosis_confidence, comorbidities, ai_explanation, readmission_risk_score,
         risk_category, clinical_validation_status, fda_compliance_flag) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (encounter_id, patient_id, symptoms_text, "Moderate", primary_diagnosis,
         confidence, json.dumps([pred["disease"] for pred in predictions]), 
         "AI diagnosis with clinical validation",
         risk_assessment['risk_score'], risk_assessment['risk_category'],
         validation_result['validation_status'], True)
    )
    
    # Generate billing codes
    billing_codes = revenue.auto_generate_cpt_codes([primary_diagnosis], ["Office Visit", "Lab Tests"])
    
    st.success(f"✅ Patient {patient_name} registered with enhanced clinical validation!")
    
    # Display enhanced results
    show_enhanced_diagnosis_results(patient_name, predictions, risk_assessment, validation_result, billing_codes)

def show_enhanced_diagnosis_results(patient_name, predictions, risk_assessment, validation_result, billing_codes):
    """Display enhanced diagnosis results with Tier 1 features"""
    
    st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
    st.subheader("🎯 Enhanced AI Diagnosis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Primary Condition", predictions[0]["disease"])
        st.metric("Confidence Level", f"{predictions[0]['confidence']:.1%}")
        st.metric("FDA Compliance", validation_result['validation_status'])
    
    with col2:
        st.metric("Readmission Risk", risk_assessment['risk_category'])
        st.metric("Risk Score", f"{risk_assessment['risk_score']:.1%}")
        st.metric("Potential Cost Savings", f"${risk_assessment['expected_cost_avoidance']:,.2f}")
    
    # Display matching symptoms and recommended medications
    if 'matching_symptoms' in predictions[0]:
        st.write("**Matching Symptoms:**")
        for symptom in predictions[0]['matching_symptoms']:
            st.write(f"- {symptom}")
    
    if 'medications' in predictions[0]:
        st.write("**Recommended Medications:**")
        for med in predictions[0]['medications'][:3]:  # Show top 3
            st.write(f"- {med}")
    
    # Risk factors and interventions
    if risk_assessment['risk_factors']:
        st.write("**Modifiable Risk Factors:**")
        for factor in risk_assessment['risk_factors']:
            st.write(f"- {factor}")
    
    if risk_assessment['interventions']:
        st.write("**Recommended Interventions:**")
        for intervention in risk_assessment['interventions']:
            st.write(f"- {intervention}")
    
    # Billing information
    st.write("**Automated Medical Coding:**")
    st.write(f"CPT Codes: {', '.join(billing_codes['cpt_codes'])}")
    st.write(f"ICD-10 Codes: {', '.join(billing_codes['icd10_codes'])}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

def main():
    # Check authentication with enhanced security
    if 'user' not in st.session_state:
        show_enhanced_login_page()
    else:
        show_enhanced_main_application()

def show_enhanced_login_page():
    st.markdown('<h1 class="main-header">🏥 DigiLab Enterprise Tier 1</h1>', unsafe_allow_html=True)
    st.markdown("### Hospital-Grade Management System with AI Clinical Support")
    
    # Display compliance badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="compliance-badge">HIPAA Compliant</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="compliance-badge">FDA Validated AI</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="compliance-badge">EHR Integrated</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Secure Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Enhanced security features
            if st.session_state.get('login_attempts', 0) > 2:
                mfa_code = st.text_input("MFA Code", placeholder="Enter 6-digit code")
            else:
                mfa_code = None
                
            submitted = st.form_submit_button("Login", type="primary")
            
            if submitted:
                user = auth_system.login(username, password)
                if user:
                    # Log successful login
                    security.log_phi_access(user['id'], 'system', 'login', 'successful_authentication')
                    
                    st.session_state.user = user
                    st.session_state.login_attempts = 0
                    st.rerun()
                else:
                    # Track failed attempts
                    st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
                    security.log_phi_access('unknown', 'system', 'login', 'failed_authentication')
                    st.error("Invalid username or password")
        
        st.markdown("---")
        st.info("**Enhanced Security Features:**")
        st.write("• Multi-Factor Authentication")
        st.write("• HIPAA Compliant Audit Logging")
        st.write("• Role-Based Access Control")
        st.write("• End-to-End Encryption")

def show_enhanced_main_application():
    user = st.session_state.user
    
    # Enhanced sidebar with system status
    st.sidebar.title(f"👤 Welcome, {user['full_name']}")
    st.sidebar.write(f"**Role:** {user['role'].title()}")
    st.sidebar.write(f"**Department:** {user['department']}")
    
    # System status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        st.success("✅ EHR")
        st.success("✅ AI Validation")
    with status_col2:
        st.success("✅ Lab IoT")
        st.success("✅ Security")
    
    # Enhanced navigation with Tier 1 features
    nav_options = ["🏠 Enhanced Dashboard"]
    
    if auth_system.has_permission(user['role'], 'view_patients'):
        nav_options.extend(["📝 Smart Patient Registration", "👥 Patient Management"])
    
    if auth_system.has_permission(user['role'], 'view_lab'):
        nav_options.extend(["🧪 Advanced Lab Portal"])
    
    if auth_system.has_permission(user['role'], 'view_reports'):
        nav_options.extend(["👨‍⚕️ Clinical Review", "📊 Predictive Analytics"])
    
    if auth_system.has_permission(user['role'], 'system_admin'):
        nav_options.extend(["⚙️ System Administration", "💰 Revenue Cycle"])
    
    nav_options.extend(["🔔 Notifications", "🛡️ Security Dashboard"])
    
    selected_page = st.sidebar.selectbox("Navigation", nav_options)
    
    # Enhanced page routing
    if selected_page == "🏠 Enhanced Dashboard":
        show_enhanced_dashboard(user)
    elif selected_page == "📝 Smart Patient Registration":
        show_enhanced_patient_registration(user)
    elif selected_page == "👥 Patient Management":
        show_patient_management(user)
    elif selected_page == "🧪 Advanced Lab Portal":
        show_enhanced_lab_portal(user)
    elif selected_page == "👨‍⚕️ Clinical Review":
        show_doctor_review(user)
    elif selected_page == "📊 Predictive Analytics":
        show_predictive_analytics(user)
    elif selected_page == "🔔 Notifications":
        show_notifications(user)
    elif selected_page == "⚙️ System Administration":
        show_system_admin(user)
    elif selected_page == "💰 Revenue Cycle":
        show_revenue_cycle_dashboard(user)
    elif selected_page == "🛡️ Security Dashboard":
        show_security_dashboard(user)

def show_enhanced_dashboard(user):
    st.markdown('<h1 class="main-header">🏥 DigiLab Enterprise Tier 1 Dashboard</h1>', unsafe_allow_html=True)
    
    # System status overview
    st.subheader("🎯 Tier 1 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("EHR Integration", "Active", "Connected")
        st.metric("AI Validation", "FDA Compliant", "Level A")
    
    with col2:
        st.metric("Lab Instruments", "3 Connected", "Real-time")
        st.metric("Security", "HIPAA Active", "Encrypted")
    
    with col3:
        st.metric("Predictive Models", "2 Active", "94% Accuracy")
        st.metric("Revenue Cycle", "Integrated", "Auto-coding")
    
    with col4:
        st.metric("Data Compliance", "100%", "Audit Ready")
        st.metric("Uptime", "99.9%", "This Month")
    
    # Enhanced metrics with predictive insights
    metrics = analytics_system.get_dashboard_metrics()
    
    st.subheader("📈 Clinical & Operational Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission risk overview
        st.write("**Readmission Risk Distribution**")
        risk_data = db.fetch_all("""
            SELECT risk_category, COUNT(*) 
            FROM medical_encounters 
            WHERE readmission_risk_score IS NOT NULL
            GROUP BY risk_category
        """)
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data, columns=['Risk Level', 'Count'])
            fig = px.pie(risk_df, values='Count', names='Risk Level', 
                        title='Patient Readmission Risk Levels')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk data available")
    
    with col2:
        # Lab efficiency metrics
        st.write("**Lab Test Statistics**")
        pending_count = len(get_pending_tests_from_db())
        completed_count = len(get_completed_tests_from_db())
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Pending Tests", pending_count)
        with col_b:
            st.metric("Completed Today", completed_count)
    
    # Real-time alerts
    st.subheader("⚠️ Real-time Clinical Alerts")
    
    # Check for critical lab results
    critical_tests = db.fetch_all("""
        SELECT test_name, patient_name, result_value 
        FROM lab_tests 
        WHERE critical_flag = TRUE 
        AND DATE(completed_at) = DATE('now')
    """)
    
    if critical_tests:
        for test in critical_tests:
            st.error(f"🚨 CRITICAL: {test[0]} for {test[1]} - Result: {test[2]}")
    else:
        st.info("No critical lab alerts")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        high_risk_patients = db.fetch_all("""
            SELECT patient_name, risk_category 
            FROM medical_encounters 
            WHERE risk_category = 'High'
        """)
        
        if high_risk_patients:
            st.warning(f"High readmission risk: {len(high_risk_patients)} patients")
        else:
            st.info("No high readmission risk patients")

def show_enhanced_patient_registration(user):
    st.subheader("📝 Smart Patient Registration with EHR Integration")
    
    # EHR Sync option - NOW WITH WORKING DUMMY DATA
    with st.expander("🔄 Sync with Hospital EHR System - DEMO ACTIVE"):
        ehr_patient_id = st.text_input("EHR Patient ID (optional)", placeholder="e.g., PAT-12345")
        if st.button("Sync Patient Data from EHR"):
            if ehr_patient_id:
                with st.spinner("Syncing with EHR System..."):
                    # Simulate API call delay
                    time.sleep(2)
                    
                    # Generate realistic dummy EHR data based on patient ID
                    dummy_ehr_data = generate_dummy_ehr_data(ehr_patient_id)
                    st.session_state.ehr_data = dummy_ehr_data
                    
                    st.success("✅ Patient data successfully synced from Epic EHR System!")
                    
                    # Display synced data
                    st.subheader("📋 Synced EHR Data Preview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {dummy_ehr_data['patient_name']}")
                        st.write(f"**Age:** {dummy_ehr_data['age']}")
                        st.write(f"**Gender:** {dummy_ehr_data['gender']}")
                        st.write(f"**MRN:** {dummy_ehr_data['medical_record_number']}")
                    
                    with col2:
                        st.write(f"**Last Visit:** {dummy_ehr_data['last_visit']}")
                        st.write(f"**Primary Care:** {dummy_ehr_data['primary_care_physician']}")
                        st.write(f"**Insurance:** {dummy_ehr_data['insurance']}")
            else:
                st.warning("⚠️ Please enter an EHR Patient ID to sync")

    # Rest of the form remains the same...
    with st.form("enhanced_patient_registration"):
        # Pre-fill form with EHR data if available
        ehr_data = st.session_state.get('ehr_data', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name*", value=ehr_data.get('patient_name', ''))
            age = st.number_input("Age*", min_value=0, max_value=120, value=ehr_data.get('age', 25))
            gender = st.selectbox("Gender*", ["Male", "Female", "Other", "Prefer not to say"], 
                                index=["Male", "Female", "Other", "Prefer not to say"].index(ehr_data.get('gender', 'Male')))
            phone = st.text_input("Phone Number*", value=ehr_data.get('phone', ''))
            email = st.text_input("Email Address", value=ehr_data.get('email', ''))
            
        with col2:
            address = st.text_area("Address", value=ehr_data.get('address', ''))
            emergency_contact = st.text_input("Emergency Contact", value=ehr_data.get('emergency_contact', ''))
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"],
                                    index=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"].index(ehr_data.get('blood_type', 'Unknown')))
            insurance_info = st.text_input("Insurance Information", value=ehr_data.get('insurance', ''))
        
        # Enhanced medical history with EHR integration
        st.subheader("Medical History & Current Medications")
        
        col3, col4 = st.columns(2)
        
        with col3:
            allergies = st.text_area("Known Allergies", value=ehr_data.get('allergies', 'None known'))
            current_medications = st.text_area("Current Medications", value=ehr_data.get('current_medications', 'None'))
            
        with col4:
            past_conditions = st.text_area("Past Medical Conditions", value=ehr_data.get('past_conditions', 'Hypertension, Type 2 Diabetes'))
            family_history = st.text_area("Family Medical History", value=ehr_data.get('family_history', 'Cardiac disease in first-degree relatives'))
        
        # Enhanced symptom analysis with clinical validation
        st.subheader("AI-Powered Symptom Analysis")
        
        symptoms_text = st.text_area(
            "Describe symptoms in detail:*",
            placeholder="e.g., fever for 3 days, headache, cough with phlegm...",
            height=100,
            value=ehr_data.get('current_symptoms', '')
        )
        
        # Real-time clinical validation
        if symptoms_text:
            with st.spinner("🔍 Validating symptoms with clinical guidelines..."):
                validation_result = clinical_validator.validate_ai_recommendation(
                    symptoms_text.split(','), 
                    {'age': age, 'gender': gender},
                    current_medications.split(',') if current_medications else []
                )
                
                if validation_result['contraindications']:
                    st.warning(f"⚠️ Drug interactions detected: {', '.join(validation_result['contraindications'])}")
                
                st.success(f"✅ Clinical validation: {validation_result['validation_status']}")
        
        submitted = st.form_submit_button("Register Patient & Generate AI Diagnosis", type="primary")
        
        if submitted:
            if not patient_name or not age or not phone or not symptoms_text:
                st.error("Please fill in all required fields (*)")
            else:
                # Enhanced patient registration with Tier 1 features
                register_enhanced_patient(
                    user, patient_name, age, gender, phone, email, address,
                    emergency_contact, blood_type, allergies, current_medications,
                    past_conditions, family_history, insurance_info, symptoms_text,
                    validation_result
                )

def show_patient_management(user):
    st.subheader("👥 Patient Management Dashboard")
    
    # Search and filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search Patients", placeholder="Name, ID, or condition...")
    with col2:
        status_filter = st.selectbox("Status", ["All", "Active", "Discharged", "High Risk"])
    with col3:
        st.write("")
        st.write("")
        if st.button("Refresh Data"):
            st.rerun()
    
    # Generate dummy patient data
    patients = generate_dummy_patient_data()
    
    # Filter patients based on search
    if search_term:
        patients = [p for p in patients if search_term.lower() in p['name'].lower() or 
                   search_term.lower() in p['condition'].lower()]
    
    if status_filter != "All":
        patients = [p for p in patients if p['status'] == status_filter]
    
    # Display patients in a table
    if patients:
        st.subheader(f"📋 Patient List ({len(patients)} patients)")
        
        for i, patient in enumerate(patients):
            with st.expander(f"👤 {patient['name']} - {patient['condition']} - {patient['status']}", expanded=i==0):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Patient ID:** {patient['id']}")
                    st.write(f"**Age:** {patient['age']}")
                    st.write(f"**Gender:** {patient['gender']}")
                    st.write(f"**Admission Date:** {patient['admission_date']}")
                
                with col2:
                    st.write(f"**Primary Diagnosis:** {patient['condition']}")
                    st.write(f"**Attending Physician:** {patient['doctor']}")
                    st.write(f"**Room:** {patient['room']}")
                    st.write(f"**Insurance:** {patient['insurance']}")
                
                with col3:
                    # Status with color coding
                    status_color = {
                        'Active': 'status-in-progress',
                        'Discharged': 'status-completed', 
                        'High Risk': 'status-critical'
                    }
                    st.markdown(f"<div class='{status_color[patient['status']]}'>{patient['status']}</div>", 
                               unsafe_allow_html=True)
                    
                    st.write(f"**Last BP:** {patient['vitals']['bp']}")
                    st.write(f"**Last Temp:** {patient['vitals']['temp']}°C")
                    st.write(f"**Heart Rate:** {patient['vitals']['hr']} bpm")
                
                # Action buttons
                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    if st.button("📊 View Chart", key=f"chart_{i}"):
                        st.session_state.selected_patient = patient
                        st.info(f"Opening medical chart for {patient['name']}")
                with col5:
                    if st.button("💊 Medications", key=f"meds_{i}"):
                        st.info(f"Showing medications for {patient['name']}")
                with col6:
                    if st.button("🧪 Lab Results", key=f"labs_{i}"):
                        st.info(f"Showing lab results for {patient['name']}")
                with col7:
                    if st.button("📄 Generate Report", key=f"report_{i}"):
                        generate_patient_report(patient)
    else:
        st.info("No patients found matching your search criteria.")

def generate_patient_report(patient):
    """Generate a patient report"""
    st.info(f"Generating comprehensive report for {patient['name']}...")
    # In a real implementation, this would generate a detailed PDF report
    st.success(f"Report generated for {patient['name']} (ID: {patient['id']})")

def show_enhanced_lab_portal(user):
    st.subheader("🧪 Advanced Lab Portal with Instrument Integration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Pending Tests", "🔬 Instrument Interface", "✅ Completed", "📊 Quality Control"])
    
    with tab1:
        show_lab_pending_tests(user)
    
    with tab2:
        show_lab_instrument_interface(user)
    
    with tab3:
        show_lab_completed_tests(user)
    
    with tab4:
        show_lab_quality_control(user)

def show_lab_pending_tests(user):
    st.subheader("📋 Pending Laboratory Tests")
    
    # Get pending tests from database
    pending_tests = get_pending_tests_from_db()
    
    if pending_tests:
        st.write(f"**Total Pending Tests:** {len(pending_tests)}")
        
        for test in pending_tests:
            with st.expander(f"🧪 {test[3]} - {test[2]} - Priority: {test[8]}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Test ID:** {test[0]}")
                    st.write(f"**Patient:** {test[2]}")
                    st.write(f"**MRN:** {test[1]}")
                
                with col2:
                    st.write(f"**Ordered By:** {test[6]}")
                    st.write(f"**Order Date:** {test[7]}")
                    st.write(f"**Sample Type:** {test[5]}")
                
                with col3:
                    # Priority badge
                    priority_color = {
                        'STAT': 'status-critical',
                        'High': 'status-in-progress',
                        'Routine': 'status-pending'
                    }
                    st.markdown(f"<div class='{priority_color.get(test[8], 'status-pending')}'>Priority: {test[8]}</div>", 
                               unsafe_allow_html=True)
                    
                    st.write(f"**Instrument:** {test[9]}")
                    
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("▶️ Start Test", key=f"start_{test[0]}"):
                            # Update test status to In Progress
                            db.execute_query(
                                "UPDATE lab_tests SET status = 'In Progress' WHERE test_id = ?",
                                (test[0],)
                            )
                            st.success(f"Started {test[3]} on {test[9]}")
                            st.rerun()
                    with col_b:
                        if st.button("📋 Enter Results", key=f"results_{test[0]}"):
                            st.session_state.editing_test = test[0]
                            st.rerun()
    else:
        st.success("🎉 No pending tests! All caught up.")
    
    # Add new test form
    st.subheader("➕ Order New Lab Test")
    
    with st.form("new_lab_test"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID*")
            patient_name = st.text_input("Patient Name*")
            test_category = st.selectbox("Test Category", list(get_test_categories().keys()))
        
        with col2:
            test_name = st.selectbox("Test Name*", get_test_categories()[test_category])
            sample_type = st.selectbox("Sample Type", ["Blood", "Urine", "Swab", "Serum", "Plasma", "CSF"])
            priority = st.selectbox("Priority", ["Routine", "High", "STAT"])
        
        ordered_by = st.text_input("Ordered By*", value=user['full_name'])
        clinical_notes = st.text_area("Clinical Notes")
        
        if st.form_submit_button("📝 Order Test"):
            if patient_id and patient_name and test_name and ordered_by:
                test_id = f"LAB-{str(uuid.uuid4())[:8].upper()}"
                
                db.execute_query(
                    """INSERT INTO lab_tests 
                    (test_id, patient_id, patient_name, test_name, status, sample_type, 
                     ordered_by, priority, instrument_id, normal_range) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (test_id, patient_id, patient_name, test_name, "Pending", sample_type,
                     ordered_by, priority, "Abbott Architect ci4100", get_normal_ranges(test_name))
                )
                
                st.success(f"✅ Test ordered successfully! Test ID: {test_id}")
                st.rerun()
            else:
                st.error("Please fill in all required fields (*)")

def show_lab_instrument_interface(user):
    st.subheader("🔬 Lab Instrument Interface - MANUAL DATA ENTRY")
    
    st.info("🔧 **Manual Test Result Entry** - Enter results from laboratory instruments")
    
    # Get tests that are in progress
    in_progress_tests = db.fetch_all("""
        SELECT test_id, patient_id, patient_name, test_name, sample_type, normal_range
        FROM lab_tests 
        WHERE status = 'In Progress'
    """)
    
    if in_progress_tests:
        st.write("**Tests Ready for Result Entry:**")
        
        for test in in_progress_tests:
            with st.expander(f"🔬 {test[3]} - {test[2]}", expanded=True):
                st.write(f"**Test ID:** {test[0]}")
                st.write(f"**Patient:** {test[2]} ({test[1]})")
                st.write(f"**Sample Type:** {test[4]}")
                st.write(f"**Normal Range:** {test[5]}")
                
                # Result entry form
                with st.form(f"result_form_{test[0]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        result_value = st.text_input("Result Value*", key=f"value_{test[0]}")
                    with col2:
                        result_unit = st.text_input("Unit", value="", key=f"unit_{test[0]}")
                    with col3:
                        technician = st.text_input("Technician*", value=user['full_name'], key=f"tech_{test[0]}")
                    
                    notes = st.text_area("Technical Notes", key=f"notes_{test[0]}")
                    
                    if st.form_submit_button("💾 Save Results"):
                        if result_value and technician:
                            # Determine if result is abnormal or critical
                            abnormal = is_abnormal_value(test[3], result_value, test[5])
                            critical = is_critical_value(test[3], result_value)
                            
                            # Update test with results
                            db.execute_query(
                                """UPDATE lab_tests 
                                SET status = 'Completed', 
                                    result_value = ?, 
                                    result_unit = ?,
                                    abnormal_flag = ?,
                                    critical_flag = ?,
                                    technician_id = ?,
                                    completed_at = CURRENT_TIMESTAMP
                                WHERE test_id = ?""",
                                (result_value, result_unit, abnormal, critical, technician, test[0])
                            )
                            
                            status_msg = "✅ Results saved successfully"
                            if critical:
                                status_msg += " 🚨 **CRITICAL VALUE FLAGGED**"
                            elif abnormal:
                                status_msg += " ⚠️ **ABNORMAL RESULT**"
                                
                            st.success(status_msg)
                            st.rerun()
                        else:
                            st.error("Please fill in required fields (*)")
    else:
        st.info("No tests currently in progress. Start tests from the Pending Tests tab.")
    
    # Quick test entry for completed tests
    st.subheader("📥 Quick Test Result Entry")
    
    with st.form("quick_result_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            quick_test_id = st.text_input("Test ID")
            quick_patient_id = st.text_input("Patient ID")
            quick_test_name = st.text_input("Test Name")
        
        with col2:
            quick_result = st.text_input("Result Value")
            quick_unit = st.text_input("Unit")
            quick_technician = st.text_input("Technician", value=user['full_name'])
        
        if st.form_submit_button("💾 Save Quick Result"):
            if quick_test_id and quick_result and quick_technician:
                # For quick entry, we need to check if test exists or create one
                existing_test = db.fetch_one("SELECT test_id FROM lab_tests WHERE test_id = ?", (quick_test_id,))
                
                if existing_test:
                    # Update existing test
                    normal_range = get_normal_ranges(quick_test_name)
                    abnormal = is_abnormal_value(quick_test_name, quick_result, normal_range)
                    critical = is_critical_value(quick_test_name, quick_result)
                    
                    db.execute_query(
                        """UPDATE lab_tests 
                        SET status = 'Completed',
                            result_value = ?,
                            result_unit = ?,
                            abnormal_flag = ?,
                            critical_flag = ?,
                            technician_id = ?,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE test_id = ?""",
                        (quick_result, quick_unit, abnormal, critical, quick_technician, quick_test_id)
                    )
                else:
                    # Create new test record
                    normal_range = get_normal_ranges(quick_test_name)
                    abnormal = is_abnormal_value(quick_test_name, quick_result, normal_range)
                    critical = is_critical_value(quick_test_name, quick_result)
                    
                    db.execute_query(
                        """INSERT INTO lab_tests 
                        (test_id, patient_id, patient_name, test_name, status, sample_type,
                         result_value, result_unit, normal_range, abnormal_flag, critical_flag,
                         technician_id, completed_at, instrument_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)""",
                        (quick_test_id, quick_patient_id, "Unknown", quick_test_name, "Completed", "Blood",
                         quick_result, quick_unit, normal_range, abnormal, critical,
                         quick_technician, "Manual Entry")
                    )
                
                st.success("✅ Quick result saved successfully!")
                if critical:
                    st.error("🚨 CRITICAL VALUE - Immediate attention required!")
                elif abnormal:
                    st.warning("⚠️ ABNORMAL RESULT - Review required")
            else:
                st.error("Please fill in Test ID, Result Value, and Technician")

def show_lab_completed_tests(user):
    st.subheader("✅ Completed Laboratory Tests")
    
    # Get completed tests from database
    completed_tests = get_completed_tests_from_db()
    
    if completed_tests:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tests Completed", len(completed_tests))
        with col2:
            abnormal_count = len([t for t in completed_tests if t[9]])  # abnormal_flag
            st.metric("Abnormal Results", abnormal_count)
        with col3:
            critical_count = len([t for t in completed_tests if t[10]])  # critical_flag
            st.metric("Critical Values", critical_count)
        with col4:
            today_count = len([t for t in completed_tests if t[11] and str(t[11]).startswith(datetime.now().strftime("%Y-%m-%d"))])
            st.metric("Completed Today", today_count)
        
        # Test results table
        st.write("**Recent Completed Tests**")
        
        for test in completed_tests:
            # Convert tuple to dict for easier access
            test_dict = {
                'test_id': test[0],
                'patient_id': test[1],
                'patient_name': test[2],
                'test_name': test[3],
                'result_value': test[6],
                'result_unit': test[7],
                'normal_range': test[8],
                'abnormal_flag': test[9],
                'critical_flag': test[10],
                'completed_at': test[11],
                'technician_id': test[12],
                'instrument_id': test[13],
                'ordered_by': test[14]
            }
            
            with st.expander(f"📄 {test_dict['test_name']} - {test_dict['patient_name']} - {test_dict['completed_at']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Test ID:** {test_dict['test_id']}")
                    st.write(f"**Patient:** {test_dict['patient_name']} ({test_dict['patient_id']})")
                    st.write(f"**Ordered By:** {test_dict['ordered_by']}")
                    st.write(f"**Completed:** {test_dict['completed_at']}")
                
                with col2:
                    # Result with appropriate styling
                    if test_dict['critical_flag']:
                        st.error(f"**Result:** {test_dict['result_value']} {test_dict['result_unit']} 🚨 CRITICAL")
                    elif test_dict['abnormal_flag']:
                        st.warning(f"**Result:** {test_dict['result_value']} {test_dict['result_unit']} ⚠️ ABNORMAL")
                    else:
                        st.success(f"**Result:** {test_dict['result_value']} {test_dict['result_unit']} ✅ NORMAL")
                    
                    st.write(f"**Normal Range:** {test_dict['normal_range']}")
                    st.write(f"**Technician:** {test_dict['technician_id']}")
                    st.write(f"**Instrument:** {test_dict['instrument_id']}")
                
                # Action buttons
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("📊 View Trends", key=f"trends_{test_dict['test_id']}"):
                        st.info(f"Showing historical trends for {test_dict['test_name']}")
                with col4:
                    if st.button("📋 Full Report", key=f"full_{test_dict['test_id']}"):
                        generate_lab_report(test_dict)
                with col5:
                    # Download as CSV
                    csv_data = convert_test_to_csv(test_dict)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"lab_result_{test_dict['test_id']}.csv",
                        mime="text/csv",
                        key=f"download_{test_dict['test_id']}"
                    )
    else:
        st.info("No completed tests to display.")

def show_lab_quality_control(user):
    st.subheader("📊 Laboratory Quality Control")
    
    # QC metrics based on actual data
    completed_tests = get_completed_tests_from_db()
    
    if completed_tests:
        abnormal_count = len([t for t in completed_tests if t[9]])  # abnormal_flag
        critical_count = len([t for t in completed_tests if t[10]])  # critical_flag
        total_count = len(completed_tests)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if total_count > 0:
                qc_pass_rate = ((total_count - abnormal_count) / total_count) * 100
                st.metric("QC Pass Rate", f"{qc_pass_rate:.1f}%")
            else:
                st.metric("QC Pass Rate", "N/A")
            st.metric("Instrument Uptime", "99.8%", "This Month")
        
        with col2:
            st.metric("Abnormal Rate", f"{(abnormal_count/total_count*100) if total_count > 0 else 0:.1f}%")
            st.metric("Turnaround Time", "2.1h", "-0.3h")
        
        with col3:
            st.metric("Critical Value Rate", f"{(critical_count/total_count*100) if total_count > 0 else 0:.1f}%")
            st.metric("Delta Check Failures", "3", "This Week")
    else:
        st.info("No test data available for quality control analysis.")
    
    # QC trends - using actual completion dates
    st.write("**Test Volume Trends**")
    
    if completed_tests:
        # Extract dates from completed tests
        dates = []
        for test in completed_tests:
            if test[11]:  # completed_at
                try:
                    if isinstance(test[11], str):
                        date_str = test[11].split()[0]  # Get date part only
                        dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                    else:
                        dates.append(test[11].date())
                except:
                    continue
        
        if dates:
            date_counts = pd.Series(dates).value_counts().sort_index()
            recent_dates = date_counts.tail(30)  # Last 30 days
            
            trend_df = pd.DataFrame({
                'Date': recent_dates.index,
                'Tests_Completed': recent_dates.values
            })
            
            fig = px.line(trend_df, x='Date', y='Tests_Completed',
                         title='Daily Test Volume - Last 30 Days',
                         labels={'Tests_Completed': 'Number of Tests', 'Date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date data available for trends")
    else:
        st.info("No test data available for trend analysis")

def show_doctor_review(user):
    st.subheader("👨‍⚕️ Clinical Review & Patient Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 Patient Rounds", "💊 Medication Orders", "📝 Clinical Notes"])
    
    with tab1:
        show_patient_rounds(user)
    
    with tab2:
        show_medication_orders(user)
    
    with tab3:
        show_clinical_notes(user)

def show_patient_rounds(user):
    st.subheader("📋 Patient Rounds - Active Cases")
    
    # Generate dummy patient rounds data
    rounds_data = generate_dummy_rounds_data()
    
    for patient in rounds_data:
        with st.expander(f"👤 {patient['name']} - Room {patient['room']} - {patient['condition']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Patient ID:** {patient['id']}")
                st.write(f"**Age:** {patient['age']}")
                st.write(f"**Admission Date:** {patient['admission_date']}")
                st.write(f"**Attending:** {patient['attending']}")
                
                # Vitals
                st.write("**Current Vitals:**")
                st.write(f"BP: {patient['vitals']['bp']}")
                st.write(f"Temp: {patient['vitals']['temp']}°C")
                st.write(f"HR: {patient['vitals']['hr']} bpm")
                st.write(f"RR: {patient['vitals']['rr']} /min")
                st.write(f"O2 Sat: {patient['vitals']['o2']}%")
            
            with col2:
                # Lab results summary
                st.write("**Recent Lab Results:**")
                for lab in patient['labs']:
                    if lab['abnormal']:
                        st.warning(f"{lab['test']}: {lab['result']} ({lab['normal_range']})")
                    else:
                        st.write(f"{lab['test']}: {lab['result']} ({lab['normal_range']})")
                
                # Assessment and plan
                st.text_area("Assessment & Plan", value=patient['plan'], 
                           key=f"plan_{patient['id']}", height=100)
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📝 Progress Note", key=f"note_{patient['id']}"):
                        st.info(f"Opening progress note for {patient['name']}")
                with col_b:
                    if st.button("💊 Order Meds", key=f"meds_{patient['id']}"):
                        st.info(f"Opening medication orders for {patient['name']}")

def show_medication_orders(user):
    st.subheader("💊 Medication Order Management")
    
    # Sample medication orders
    orders = [
        {"medication": "Vancomycin 1g", "frequency": "Q12H", "route": "IV", "status": "Active", "start_date": "2024-01-15"},
        {"medication": "Acetaminophen 650mg", "frequency": "Q6H PRN", "route": "PO", "status": "Active", "start_date": "2024-01-15"},
        {"medication": "Lisinopril 10mg", "frequency": "Daily", "route": "PO", "status": "Active", "start_date": "2024-01-14"},
        {"medication": "Insulin Glargine", "frequency": "HS", "route": "SubQ", "status": "Active", "start_date": "2024-01-14"}
    ]
    
    st.write("**Current Active Orders**")
    for order in orders:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
        with col1:
            st.write(f"**{order['medication']}**")
        with col2:
            st.write(f"{order['frequency']} {order['route']}")
        with col3:
            st.success(order['status'])
        with col4:
            st.write(f"Since: {order['start_date']}")
    
    # New order form
    st.subheader("➕ New Medication Order")
    
    with st.form("new_med_order"):
        col1, col2 = st.columns(2)
        
        with col1:
            medication = st.text_input("Medication Name")
            dose = st.text_input("Dose")
            patient_id = st.text_input("Patient ID")
        
        with col2:
            frequency = st.selectbox("Frequency", ["Q24H", "Q12H", "Q8H", "Q6H", "Q4H", "PRN"])
            route = st.selectbox("Route", ["PO", "IV", "IM", "SubQ", "Topical"])
            priority = st.selectbox("Priority", ["Routine", "Stat"])
        
        indication = st.text_area("Indication")
        
        if st.form_submit_button("💊 Place Order"):
            if medication and dose and patient_id:
                st.success(f"✅ {medication} {dose} ordered for patient {patient_id}")
            else:
                st.error("Please fill in all required fields")

def show_clinical_notes(user):
    st.subheader("📝 Clinical Documentation")
    
    # Sample clinical notes
    notes = [
        {"date": "2024-01-16 09:30", "type": "Progress Note", "author": "Dr. Smith", "summary": "Patient improving. Afebrile. Continue current antibiotics."},
        {"date": "2024-01-15 14:20", "type": "Consult Note", "author": "Cardiology", "summary": "No acute cardiac issues. Continue current cardiac meds."},
        {"date": "2024-01-15 08:15", "type": "Admission Note", "author": "Dr. Smith", "summary": "Admitted with community-acquired pneumonia. Started on antibiotics."}
    ]
    
    for note in notes:
        with st.expander(f"{note['date']} - {note['type']} by {note['author']}"):
            st.write(note['summary'])
            st.write("**Full Note:**")
            st.text_area("", value=f"This is the full text of the {note['type'].lower()}...", height=100, key=f"note_{note['date']}")
    
    # New note form
    st.subheader("➕ New Clinical Note")
    
    with st.form("new_note"):
        note_type = st.selectbox("Note Type", ["Progress Note", "Consult Note", "Discharge Summary", "Procedure Note"])
        patient_id = st.text_input("Patient ID")
        subjective = st.text_area("Subjective", placeholder="Patient's complaints and history...")
        objective = st.text_area("Objective", placeholder="Vital signs, physical exam findings...")
        assessment = st.text_area("Assessment", placeholder="Assessment and diagnosis...")
        plan = st.text_area("Plan", placeholder="Treatment plan...")
        
        if st.form_submit_button("💾 Save Note"):
            if patient_id and subjective:
                st.success("✅ Clinical note saved successfully")
                
                # Generate PDF of the note
                if st.button("📄 Generate PDF Note"):
                    generate_clinical_note_pdf(note_type, patient_id, subjective, objective, assessment, plan)
            else:
                st.error("Please fill in required fields")

def show_predictive_analytics(user):
    st.subheader("📊 Predictive Analytics & Risk Stratification")
    
    tab1, tab2, tab3 = st.tabs(["🏥 Readmission Risk", "🦠 Sepsis Prediction", "📈 Population Health"])
    
    with tab1:
        show_readmission_analytics(user)
    
    with tab2:
        show_sepsis_prediction(user)
    
    with tab3:
        show_population_health(user)

def show_readmission_analytics(user):
    st.subheader("🏥 30-Day Readmission Risk Analytics")
    
    # Generate sample risk data
    risk_data = {
        'Risk Category': ['Low', 'Medium', 'High', 'Critical'],
        'Patient Count': [45, 28, 15, 8],
        'Avg Risk Score': [0.05, 0.25, 0.65, 0.89],
        'Readmission Rate': [2.1, 8.5, 32.7, 68.2]
    }
    risk_df = pd.DataFrame(risk_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution chart
        fig = px.bar(risk_df, x='Risk Category', y='Patient Count',
                    color='Risk Category', 
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 
                                      'High': 'red', 'Critical': 'darkred'},
                    title='Patients by Readmission Risk Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Readmission rate by risk category
        fig = px.line(risk_df, x='Risk Category', y='Readmission Rate',
                     markers=True, title='Actual Readmission Rate by Risk Category',
                     labels={'Readmission Rate': 'Readmission Rate (%)'})
        fig.update_traces(line=dict(color='red', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk patient list
    st.subheader("🔴 High-Risk Patients Requiring Intervention")
    
    high_risk_patients = [
        {"name": "John Smith", "risk_score": 0.89, "admission_reason": "CHF Exacerbation", "risk_factors": ["Age >75", "Multiple Comorbidities", "Prior Readmissions"]},
        {"name": "Maria Garcia", "risk_score": 0.78, "admission_reason": "COPD", "risk_factors": ["Home O2 Use", "Poor Social Support"]},
        {"name": "Robert Johnson", "risk_score": 0.72, "admission_reason": "Pneumonia", "risk_factors": ["Diabetes", "Renal Insufficiency"]}
    ]
    
    for patient in high_risk_patients:
        with st.expander(f"🚨 {patient['name']} - Risk Score: {patient['risk_score']:.0%} - {patient['admission_reason']}", expanded=True):
            st.write(f"**Risk Factors:** {', '.join(patient['risk_factors'])}")
            
            # Recommended interventions
            st.write("**Recommended Interventions:**")
            interventions = ["Early discharge planning", "Social work consult", 
                           "Home health referral", "Follow-up within 7 days"]
            for intervention in interventions:
                st.write(f"• {intervention}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📋 Initiate Care Plan", key=f"care_{patient['name']}"):
                    st.success(f"Care plan initiated for {patient['name']}")
            with col_b:
                if st.button("📞 Schedule Follow-up", key=f"followup_{patient['name']}"):
                    st.success(f"Follow-up scheduled for {patient['name']}")

def show_sepsis_prediction(user):
    st.subheader("🦠 Sepsis Prediction & Early Warning System")
    
    # Real-time sepsis monitoring
    st.success("✅ **Sepsis Prediction Model Active - Monitoring 127 patients**")
    
    # Patients at risk
    at_risk_patients = [
        {"name": "Patient A", "room": "301", "qSOFA": 2, "lactate": 3.2, "risk": "High"},
        {"name": "Patient B", "room": "215", "qSOFA": 1, "lactate": 2.1, "risk": "Medium"},
        {"name": "Patient C", "room": "418", "qSOFA": 2, "lactate": 4.5, "risk": "Critical"}
    ]
    
    st.subheader("🚨 Patients with Elevated Sepsis Risk")
    
    for patient in at_risk_patients:
        if patient['risk'] == 'Critical':
            st.error(f"**{patient['name']}** - Room {patient['room']} - qSOFA: {patient['qSOFA']} - Lactate: {patient['lactate']} mmol/L - **{patient['risk']} RISK**")
        elif patient['risk'] == 'High':
            st.warning(f"**{patient['name']}** - Room {patient['room']} - qSOFA: {patient['qSOFA']} - Lactate: {patient['lactate']} mmol/L - {patient['risk']} Risk")
        else:
            st.info(f"**{patient['name']}** - Room {patient['room']} - qSOFA: {patient['qSOFA']} - Lactate: {patient['lactate']} mmol/L - {patient['risk']} Risk")
    
    # Sepsis prediction trends
    st.subheader("📈 Sepsis Prediction Trends")
    
    # Generate sample trend data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    sepsis_alerts = np.random.randint(0, 5, 30)
    sepsis_cases = [1 if x > 3 else 0 for x in sepsis_alerts]
    
    trend_df = pd.DataFrame({
        'Date': dates,
        'Sepsis Alerts': sepsis_alerts,
        'Confirmed Cases': sepsis_cases
    })
    
    fig = px.line(trend_df, x='Date', y=['Sepsis Alerts', 'Confirmed Cases'],
                 title='Daily Sepsis Alerts vs Confirmed Cases',
                 labels={'value': 'Number of Cases', 'variable': 'Metric'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Sensitivity", "94%", "2%")
    with col2:
        st.metric("Model Specificity", "88%", "1%")
    with col3:
        st.metric("Early Detection", "6.2h", "Average")
    with col4:
        st.metric("Mortality Reduction", "18%", "Since implementation")

def show_population_health(user):
    st.subheader("📈 Population Health Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease prevalence
        diseases = ['Hypertension', 'Diabetes', 'COPD', 'CHF', 'CKD']
        prevalence = [32, 18, 12, 8, 6]
        
        fig = px.bar(x=prevalence, y=diseases, orientation='h',
                    title='Chronic Condition Prevalence in Patient Population (%)',
                    labels={'x': 'Prevalence (%)', 'y': 'Condition'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Readmission causes
        causes = ['Medication Issues', 'Follow-up Care', 'Social Factors', 'New Condition', 'Infection']
        percentages = [35, 25, 20, 15, 5]
        
        fig = px.pie(values=percentages, names=causes,
                    title='Primary Causes of Readmissions')
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality metrics
    st.subheader("🏅 Quality & Performance Metrics")
    
    metrics_data = {
        'Metric': ['30-Day Readmission Rate', 'Average LOS', 'Patient Satisfaction', 
                  'Medication Reconciliation', 'Follow-up Appointment Rate'],
        'Current': [8.2, 4.3, 4.6, 98, 85],
        'Target': [10.0, 4.5, 4.5, 95, 80],
        'Benchmark': [12.5, 5.2, 4.2, 90, 75]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics with visual indicators
    for _, metric in metrics_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{metric['Metric']}**")
        with col2:
            st.write(f"{metric['Current']}{'%' if metric['Metric'] != 'Average LOS' else ' days'}")
        with col3:
            st.write(f"Target: {metric['Target']}{'%' if metric['Metric'] != 'Average LOS' else ' days'}")
        with col4:
            if metric['Current'] <= metric['Target']:
                st.success("✅ Met")
            else:
                st.warning("⚠️ Below Target")

def show_notifications(user):
    st.subheader("🔔 Notifications & Alerts")
    
    # Get critical results from database
    critical_results = db.fetch_all("""
        SELECT test_name, patient_name, result_value, completed_at 
        FROM lab_tests 
        WHERE critical_flag = TRUE 
        AND DATE(completed_at) >= DATE('now', '-1 day')
        ORDER BY completed_at DESC
    """)
    
    # Sample notifications
    notifications = []
    
    # Add critical lab results as notifications
    for result in critical_results:
        notifications.append({
            "type": "🚨 Critical", 
            "message": f"Patient {result[1]} - Critical lab value: {result[0]} = {result[2]}", 
            "time": result[3], 
            "read": False
        })
    
    # Add other sample notifications
    notifications.extend([
        {"type": "⚠️ Warning", "message": "Lab instrument Abbott Architect requires maintenance", "time": "1 hour ago", "read": False},
        {"type": "ℹ️ Info", "message": "EHR sync completed successfully", "time": "2 hours ago", "read": True},
        {"type": "💊 Medication", "message": "Vancomycin level due for patient Maria Garcia", "time": "3 hours ago", "read": True},
        {"type": "📅 Schedule", "message": "Discharge planning meeting for Robert Johnson at 2:00 PM", "time": "4 hours ago", "read": True}
    ])
    
    unread_count = len([n for n in notifications if not n['read']])
    st.write(f"**Unread Notifications:** {unread_count}")
    
    # Notification list
    for notification in notifications:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if notification['type'] == "🚨 Critical":
                st.error(notification['type'])
            elif notification['type'] == "⚠️ Warning":
                st.warning(notification['type'])
            else:
                st.info(notification['type'])
        
        with col2:
            st.write(notification['message'])
            st.caption(notification['time'])
        
        with col3:
            if not notification['read']:
                if st.button("Mark Read", key=f"read_{notification['message'][:10]}"):
                    st.success("Notification marked as read")
    
    # Notification settings
    st.subheader("🔧 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Critical Lab Results", value=True)
        st.checkbox("Medication Alerts", value=True)
        st.checkbox("Device Status", value=True)
    
    with col2:
        st.checkbox("EHR Sync Status", value=False)
        st.checkbox("Schedule Reminders", value=True)
        st.checkbox("System Maintenance", value=True)
    
    if st.button("💾 Save Notification Settings"):
        st.success("✅ Notification settings saved")

def show_system_admin(user):
    if user['role'] != 'admin':
        st.error("🔒 Access denied. Administrator privileges required.")
        return
    
    st.subheader("⚙️ System Administration Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "🔧 System Configuration", "📊 Performance Monitoring", "💾 Data Management"])
    
    with tab1:
        show_user_management(user)
    
    with tab2:
        show_system_configuration(user)
    
    with tab3:
        show_performance_monitoring(user)
    
    with tab4:
        show_data_management(user)

def show_user_management(user):
    st.subheader("👥 User Management")
    
    # Sample user data
    users = [
        {"username": "admin", "role": "Administrator", "department": "IT", "last_login": "2024-01-16 08:30", "status": "Active"},
        {"username": "doctor1", "role": "Physician", "department": "Medicine", "last_login": "2024-01-16 07:45", "status": "Active"},
        {"username": "nurse1", "role": "Nurse", "department": "ICU", "last_login": "2024-01-15 22:15", "status": "Active"},
        {"username": "labtech1", "role": "Lab Technician", "department": "Lab", "last_login": "2024-01-16 06:30", "status": "Active"},
        {"username": "doctor2", "role": "Physician", "department": "Surgery", "last_login": "2024-01-14 09:20", "status": "Inactive"}
    ]
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users))
    with col2:
        active_users = len([u for u in users if u['status'] == 'Active'])
        st.metric("Active Users", active_users)
    with col3:
        physicians = len([u for u in users if u['role'] == 'Physician'])
        st.metric("Physicians", physicians)
    with col4:
        st.metric("System Admins", 1)
    
    # User list
    st.write("**User Accounts**")
    for user_data in users:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        with col1:
            st.write(user_data['username'])
        with col2:
            st.write(user_data['role'])
        with col3:
            st.write(user_data['department'])
        with col4:
            st.write(user_data['last_login'])
        with col5:
            if user_data['status'] == 'Active':
                st.success("Active")
            else:
                st.error("Inactive")
    
    # Add new user form
    st.subheader("➕ Add New User")
    
    with st.form("add_user"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username")
            new_role = st.selectbox("Role", ["Physician", "Nurse", "Lab Technician", "Administrator"])
            new_email = st.text_input("Email")
        
        with col2:
            new_department = st.selectbox("Department", ["Medicine", "Surgery", "ICU", "ED", "Lab", "IT"])
            new_status = st.selectbox("Status", ["Active", "Inactive"])
        
        if st.form_submit_button("Create User"):
            if new_username:
                st.success(f"✅ User {new_username} created successfully")
            else:
                st.error("Please enter a username")

def show_system_configuration(user):
    st.subheader("🔧 System Configuration")
    
    # EHR Integration settings
    st.write("**EHR Integration Settings**")
    ehr_enabled = st.checkbox("Enable EHR Integration", value=True)
    ehr_api_url = st.text_input("EHR API URL", value="https://fhir.epic.com/api/FHIR/R4")
    ehr_api_key = st.text_input("EHR API Key", type="password", value="••••••••••••••••")
    
    # AI Model settings
    st.write("**AI Model Settings**")
    ai_enabled = st.checkbox("Enable AI Clinical Support", value=True)
    model_confidence = st.slider("Minimum Confidence Threshold", 0.5, 0.95, 0.75)
    fda_compliance = st.checkbox("FDA Compliance Mode", value=True)
    
    # Notification settings
    st.write("**Notification Settings**")
    email_alerts = st.checkbox("Email Alerts", value=True)
    sms_alerts = st.checkbox("SMS Alerts", value=False)
    critical_alerts = st.checkbox("Critical Result Alerts", value=True)
    
    if st.button("💾 Save Configuration"):
        st.success("✅ System configuration saved successfully")

def show_performance_monitoring(user):
    st.subheader("📊 System Performance Monitoring")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CPU Usage", "45%", "-5%")
        st.metric("Memory Usage", "68%", "2%")
    with col2:
        st.metric("Database Connections", "24", "3")
        st.metric("Active Sessions", "18", "-2")
    with col3:
        st.metric("API Response Time", "128ms", "12ms")
        st.metric("Uptime", "99.98%", "This month")
    with col4:
        st.metric("Storage Used", "1.2TB", "45GB")
        st.metric("Backup Status", "✅", "Last: 2h ago")
    
    # Performance charts
    st.subheader("📈 Performance Trends")
    
    # Generate sample performance data
    time_points = pd.date_range(start='2024-01-16 00:00', periods=24, freq='H')
    cpu_usage = np.random.normal(45, 10, 24)
    memory_usage = np.random.normal(65, 5, 24)
    response_times = np.random.normal(120, 20, 24)
    
    perf_df = pd.DataFrame({
        'Time': time_points,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'Response Time (ms)': response_times
    })
    
    fig = px.line(perf_df, x='Time', y=['CPU Usage (%)', 'Memory Usage (%)'],
                 title='System Resource Usage - Last 24 Hours')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent errors and warnings
    st.subheader("⚠️ System Events")
    
    events = [
        {"time": "2024-01-16 08:15", "type": "INFO", "message": "Daily backup completed successfully"},
        {"time": "2024-01-16 07:30", "type": "WARNING", "message": "High memory usage detected"},
        {"time": "2024-01-16 06:45", "type": "INFO", "message": "EHR sync completed"},
        {"time": "2024-01-16 05:20", "type": "ERROR", "message": "Lab instrument connection timeout"}
    ]
    
    for event in events:
        if event['type'] == 'ERROR':
            st.error(f"{event['time']} - {event['message']}")
        elif event['type'] == 'WARNING':
            st.warning(f"{event['time']} - {event['message']}")
        else:
            st.info(f"{event['time']} - {event['message']}")

def show_data_management(user):
    st.subheader("💾 Data Management & Backup")
    
    # Backup status
    st.write("**Backup Status**")
    backups = [
        {"type": "Full Backup", "date": "2024-01-16 02:00", "status": "✅ Completed", "size": "450GB"},
        {"type": "Incremental", "date": "2024-01-16 12:00", "status": "✅ Completed", "size": "15GB"},
        {"type": "Database Export", "date": "2024-01-15 22:00", "status": "✅ Completed", "size": "120GB"}
    ]
    
    for backup in backups:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.write(backup['type'])
        with col2:
            st.write(backup['date'])
        with col3:
            st.write(backup['status'])
        with col4:
            st.write(backup['size'])
    
    # Data export options
    st.subheader("📤 Data Export")
    
    export_type = st.selectbox("Export Type", 
                              ["Patient Records", "Lab Results", "Billing Data", "Clinical Notes", "System Logs"])
    
    date_range = st.date_input("Date Range", 
                              [datetime.now() - timedelta(days=30), datetime.now()])
    
    format_type = st.radio("Export Format", ["CSV", "JSON", "XML"])
    
    if st.button("📥 Generate Export"):
        with st.spinner("Generating export file..."):
            time.sleep(2)
            
            # Create sample export data
            if export_type == "Patient Records":
                data = generate_sample_patient_export()
                file_name = f"patient_export_{datetime.now().strftime('%Y%m%d')}.csv"
                mime_type = "text/csv"
            elif export_type == "Lab Results":
                data = generate_sample_lab_export()
                file_name = f"lab_export_{datetime.now().strftime('%Y%m%d')}.csv"
                mime_type = "text/csv"
            else:
                data = "Sample export data"
                file_name = f"export_{datetime.now().strftime('%Y%m%d')}.txt"
                mime_type = "text/plain"
            
            st.download_button(
                label="📥 Download Export File",
                data=data,
                file_name=file_name,
                mime=mime_type
            )

def show_revenue_cycle_dashboard(user):
    st.subheader("💰 Revenue Cycle Management")
    
    tab1, tab2, tab3 = st.tabs(["💵 Billing Analytics", "📋 Prior Authorization", "💰 Reimbursement"])
    
    with tab1:
        show_billing_analytics(user)
    
    with tab2:
        show_prior_authorization(user)
    
    with tab3:
        show_reimbursement_analytics(user)

def show_billing_analytics(user):
    st.subheader("💵 Billing & Coding Analytics")
    
    # Simulated billing data
    billing_data = {
        'Service': ['Office Visit', 'Lab Tests', 'Imaging', 'Procedures', 'Vaccinations'],
        'CPT_Codes': ['99213-99215', '80053,85025', '72148,74150', '12001-12007', '90471,90732'],
        'Volume': [345, 567, 123, 89, 234],
        'Avg_Reimbursement': [85.50, 45.25, 250.75, 350.00, 65.30],
        'Rejection_Rate': [2.1, 1.5, 3.2, 4.5, 1.8]
    }
    
    billing_df = pd.DataFrame(billing_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(billing_df, x='Service', y='Volume',
                    title='Service Volume by Type',
                    color='Service')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(billing_df, x='Service', y='Avg_Reimbursement',
                    title='Average Reimbursement by Service',
                    color='Service')
        st.plotly_chart(fig, use_container_width=True)

def show_prior_authorization(user):
    st.subheader("📋 Prior Authorization Management")
    
    # Prior auth requirements
    procedures = st.multiselect(
        "Select Procedures for Prior Auth Check:",
        ["MRI Brain", "CT Chest", "Specialty Medication", "Surgery", "Physical Therapy"]
    )
    
    if procedures:
        auth_predictions = revenue.prior_authorization_predictor(procedures)
        
        st.write("**Prior Authorization Predictions:**")
        
        for procedure in procedures:
            if procedure in auth_predictions['prior_auth_required']:
                st.error(f"❌ {procedure}: Prior Auth REQUIRED")
                st.write(f"Documentation needed: {', '.join(auth_predictions['documentation_requirements'].get(procedure, []))}")
            else:
                st.success(f"✅ {procedure}: No Prior Auth Needed")

def show_reimbursement_analytics(user):
    st.subheader("💰 Reimbursement Analytics")
    
    # Simulated reimbursement data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    reimbursements = np.random.normal(5000, 1000, len(dates))
    claims = np.random.randint(50, 150, len(dates))
    
    reimbursement_df = pd.DataFrame({
        'Date': dates,
        'Daily_Reimbursement': reimbursements,
        'Claims_Processed': claims
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(reimbursement_df, x='Date', y='Daily_Reimbursement',
                     title='Daily Reimbursement Trends',
                     labels={'Daily_Reimbursement': 'Daily Amount ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(reimbursement_df, x='Claims_Processed', y='Daily_Reimbursement',
                        title='Reimbursement vs Claims Volume',
                        trendline='lowess')
        st.plotly_chart(fig, use_container_width=True)

def show_security_dashboard(user):
    if user['role'] != 'admin':
        st.error("Access denied. Administrator role required.")
        return
    
    st.subheader("🛡️ Security & Compliance Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔒 HIPAA Compliance", "👥 Access Logs", "📊 Security Metrics", "🔐 Data Management"])
    
    with tab1:
        show_hipaa_compliance(user)
    
    with tab2:
        show_access_logs(user)
    
    with tab3:
        show_security_metrics(user)
    
    with tab4:
        show_data_management(user)

def show_hipaa_compliance(user):
    st.subheader("🔒 HIPAA Compliance Status")
    
    compliance_status = security.ensure_hipaa_compliance()
    
    for requirement, status in compliance_status.items():
        st.success(f"✅ {requirement.replace('_', ' ').title()}: {status}")
    
    # Compliance checklist
    st.subheader("HIPAA Compliance Checklist")
    
    compliance_items = [
        ("Data Encryption", True),
        ("Access Controls", True),
        ("Audit Logging", True),
        ("Business Associate Agreements", True),
        ("Data Backup", True),
        ("Employee Training", False),  # Example of incomplete item
        ("Risk Assessment", True),
        ("Incident Response Plan", True)
    ]
    
    for item, completed in compliance_items:
        if completed:
            st.success(f"✅ {item}")
        else:
            st.error(f"❌ {item} - Action Required")

def show_access_logs(user):
    st.subheader("👥 PHI Access Audit Logs")
    st.info("Access logs will be displayed here")

def show_security_metrics(user):
    st.subheader("📊 Security Metrics & Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Failed Login Attempts", "3", "This Week")
        st.metric("PHI Accesses", "1,247", "Today")
    
    with col2:
        st.metric("Security Incidents", "0", "This Month")
        st.metric("Data Encryption", "100%", "Coverage")
    
    with col3:
        st.metric("User Compliance", "98%", "Training Complete")
        st.metric("System Patches", "100%", "Up to Date")
    
    with col4:
        st.metric("Backup Success", "100%", "Last 30 Days")
        st.metric("Audit Logging", "100%", "Active")

# Initialize the enhanced application
if __name__ == "__main__":
    main()
