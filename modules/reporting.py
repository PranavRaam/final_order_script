"""
Reporting Module - Comprehensive Analytics and Summary Generation
Handles detailed analytics, reporting, and summary generation for parsing results
"""

import os
from typing import List, Dict, Any
from datetime import datetime

# Import data structures
from .data_structures import ParsedResult, PatientData, OrderData

def save_comprehensive_summary(results: List[ParsedResult], output_path: str):
    """Save comprehensive parsing summary with detailed analytics"""
    total_docs = len(results)
    successful = [r for r in results if r.status == 'parsed']
    failed = [r for r in results if r.status == 'failed']
    
    # Method breakdown
    structured_count = len([r for r in successful if r.source == 'structured'])
    llm_count = len([r for r in successful if r.source == 'llm'])
    
    # Quality metrics
    avg_confidence = sum(r.confidence_score for r in successful) / max(len(successful), 1)
    avg_completeness = sum(r.completeness_score for r in successful) / max(len(successful), 1)
    avg_processing_time = sum(r.processing_time for r in results) / max(total_docs, 1)
    
    # Confidence distribution
    high_confidence = len([r for r in successful if r.confidence_score >= 0.8])
    medium_confidence = len([r for r in successful if 0.5 <= r.confidence_score < 0.8])
    low_confidence = len([r for r in successful if r.confidence_score < 0.5])
    
    # Completeness distribution
    high_completeness = len([r for r in successful if r.completeness_score >= 0.7])
    medium_completeness = len([r for r in successful if 0.4 <= r.completeness_score < 0.7])
    low_completeness = len([r for r in successful if r.completeness_score < 0.4])
    
    # Field extraction analysis for successful parses
    patient_field_stats = {}
    order_field_stats = {}
    
    if successful:
        # Analyze patient field extraction rates
        patient_fields = vars(PatientData()).keys()
        for field in patient_fields:
            filled_count = sum(1 for r in successful if getattr(r.patient_data, field, "").strip())
            patient_field_stats[field] = {
                'filled': filled_count,
                'percentage': (filled_count / len(successful)) * 100
            }
        
        # Analyze order field extraction rates
        order_fields = vars(OrderData()).keys()
        for field in order_fields:
            if field == 'episode_diagnoses':
                filled_count = sum(1 for r in successful if len(r.order_data.episode_diagnoses) > 0)
            else:
                filled_count = sum(1 for r in successful if getattr(r.order_data, field, "").strip())
            order_field_stats[field] = {
                'filled': filled_count,
                'percentage': (filled_count / len(successful)) * 100
            }
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_content = f"""
Enhanced Data Parser - Comprehensive Summary Report
Generated: {timestamp}

OVERALL STATISTICS
==================
Total Documents Processed: {total_docs}
Successfully Parsed: {len(successful)} ({(len(successful)/max(total_docs,1)*100):.1f}%)
Failed to Parse: {len(failed)} ({(len(failed)/max(total_docs,1)*100):.1f}%)
Average Processing Time: {avg_processing_time:.2f}s per document

PARSING METHOD BREAKDOWN
=======================
Structured Parser: {structured_count} documents ({(structured_count/max(len(successful),1)*100):.1f}% of successful)
LLM Fallback: {llm_count} documents ({(llm_count/max(len(successful),1)*100):.1f}% of successful)

QUALITY METRICS
===============
Average Confidence Score: {avg_confidence:.3f}
Average Completeness Score: {avg_completeness:.3f}

Confidence Distribution:
  High (≥0.8): {high_confidence} documents ({(high_confidence/max(len(successful),1)*100):.1f}%)
  Medium (0.5-0.8): {medium_confidence} documents ({(medium_confidence/max(len(successful),1)*100):.1f}%)
  Low (<0.5): {low_confidence} documents ({(low_confidence/max(len(successful),1)*100):.1f}%)

Completeness Distribution:
  High (≥0.7): {high_completeness} documents ({(high_completeness/max(len(successful),1)*100):.1f}%)
  Medium (0.4-0.7): {medium_completeness} documents ({(medium_completeness/max(len(successful),1)*100):.1f}%)
  Low (<0.4): {low_completeness} documents ({(low_completeness/max(len(successful),1)*100):.1f}%)

PATIENT DATA FIELD EXTRACTION RATES
==================================="""
    
    if patient_field_stats:
        for field, stats in sorted(patient_field_stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
            summary_content += f"\n{field:30}: {stats['filled']:3d}/{len(successful):3d} ({stats['percentage']:5.1f}%)"
    
    summary_content += f"""

ORDER DATA FIELD EXTRACTION RATES
================================="""
    
    if order_field_stats:
        for field, stats in sorted(order_field_stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
            summary_content += f"\n{field:30}: {stats['filled']:3d}/{len(successful):3d} ({stats['percentage']:5.1f}%)"
    
    if failed:
        summary_content += f"""

FAILURE ANALYSIS
================
Total Failures: {len(failed)}

Common Failure Reasons:"""
        
        error_counts = {}
        for result in failed:
            error_type = result.error.split(':')[0] if ':' in result.error else result.error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            summary_content += f"\n  {error}: {count} documents"
    
    summary_content += f"""

EXTRACTION METHOD DETAILS
========================="""
    
    method_counts = {}
    for result in successful:
        method_counts[result.extraction_method] = method_counts.get(result.extraction_method, 0) + 1
    
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        summary_content += f"\n{method}: {count} documents"
    
    summary_content += f"""

PERFORMANCE ANALYSIS
====================
Total Processing Time: {sum(r.processing_time for r in results):.2f}s
Average per Document: {avg_processing_time:.2f}s
Fastest Document: {min(r.processing_time for r in results if r.processing_time > 0):.2f}s
Slowest Document: {max(r.processing_time for r in results):.2f}s

RECOMMENDATIONS
==============="""
    
    if len(successful) > 0:
        if avg_confidence < 0.6:
            summary_content += "\n⚠️  Average confidence is low - consider improving text extraction quality"
        if avg_completeness < 0.5:
            summary_content += "\n⚠️  Average completeness is low - review field extraction patterns"
        if structured_count / max(len(successful), 1) < 0.7:
            summary_content += "\n💡 Many documents required LLM fallback - consider enhancing structured parser"
        if len(failed) / max(total_docs, 1) > 0.2:
            summary_content += "\n🔍 High failure rate - review text extraction and preprocessing"
        if avg_processing_time > 10.0:
            summary_content += "\n⚡ Processing time is high - consider optimization"
        
        if avg_confidence >= 0.8 and avg_completeness >= 0.7:
            summary_content += "\n✅ Excellent extraction quality achieved!"
    else:
        summary_content += "\n❌ No successful extractions - review extraction pipeline"
    
    # Field-specific recommendations
    if patient_field_stats:
        low_extraction_fields = [field for field, stats in patient_field_stats.items() 
                               if stats['percentage'] < 30 and field in ['patient_fname', 'patient_lname', 'dob']]
        if low_extraction_fields:
            summary_content += f"\n🎯 Critical patient fields with low extraction: {', '.join(low_extraction_fields)}"
    
    if order_field_stats:
        low_extraction_fields = [field for field, stats in order_field_stats.items() 
                               if stats['percentage'] < 30 and field in ['order_date', 'physician_name']]
        if low_extraction_fields:
            summary_content += f"\n🎯 Critical order fields with low extraction: {', '.join(low_extraction_fields)}"
    
    summary_content += f"""

TECHNICAL DETAILS
=================
Documents per Method:
  - Enhanced Structured: {structured_count}
  - Enhanced LLM: {llm_count}  
  - Failed: {len(failed)}

Field Coverage:
  - Patient fields tracked: {len(patient_field_stats)}
  - Order fields tracked: {len(order_field_stats)}
  - Total fields extracted: {len(patient_field_stats) + len(order_field_stats)}

Quality Thresholds:
  - High confidence: ≥0.8
  - Medium confidence: 0.5-0.8
  - Low confidence: <0.5
  - High completeness: ≥0.7
  - Medium completeness: 0.4-0.7
  - Low completeness: <0.4

Report Generated: {timestamp}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

def generate_field_analysis_report(results: List[ParsedResult], output_path: str):
    """Generate detailed field-by-field analysis report"""
    successful = [r for r in results if r.status == 'parsed']
    
    if not successful:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""
Field Extraction Analysis Report
Generated: {timestamp}

DETAILED FIELD ANALYSIS
=======================

This report provides detailed statistics for each field extracted from {len(successful)} successfully parsed documents.

PATIENT FIELD ANALYSIS
======================
"""
    
    # Patient field analysis
    patient_fields = vars(PatientData()).keys()
    for field in sorted(patient_fields):
        filled_count = sum(1 for r in successful if getattr(r.patient_data, field, "").strip())
        percentage = (filled_count / len(successful)) * 100
        
        # Get sample values (first 3 non-empty)
        sample_values = []
        for r in successful:
            value = getattr(r.patient_data, field, "").strip()
            if value and len(sample_values) < 3:
                sample_values.append(value)
        
        report_content += f"""
{field.upper()}:
  Extraction Rate: {percentage:.1f}% ({filled_count}/{len(successful)})
  Sample Values: {', '.join(sample_values[:3]) if sample_values else 'None'}
  Status: {'✅ Good' if percentage >= 70 else '⚠️ Fair' if percentage >= 30 else '❌ Poor'}
"""
    
    report_content += f"""

ORDER FIELD ANALYSIS
====================
"""
    
    # Order field analysis
    order_fields = vars(OrderData()).keys()
    for field in sorted(order_fields):
        if field == 'episode_diagnoses':
            filled_count = sum(1 for r in successful if len(r.order_data.episode_diagnoses) > 0)
            sample_values = []
            for r in successful:
                if r.order_data.episode_diagnoses and len(sample_values) < 3:
                    sample_values.append(r.order_data.episode_diagnoses[0].diagnosis_description)
        else:
            filled_count = sum(1 for r in successful if getattr(r.order_data, field, "").strip())
            sample_values = []
            for r in successful:
                value = getattr(r.order_data, field, "").strip()
                if value and len(sample_values) < 3:
                    sample_values.append(value)
        
        percentage = (filled_count / len(successful)) * 100
        
        report_content += f"""
{field.upper()}:
  Extraction Rate: {percentage:.1f}% ({filled_count}/{len(successful)})
  Sample Values: {', '.join(sample_values[:3]) if sample_values else 'None'}
  Status: {'✅ Good' if percentage >= 70 else '⚠️ Fair' if percentage >= 30 else '❌ Poor'}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

# ------------- Completeness Report -------------

def write_completeness_report(results: List[ParsedResult], output_path: str):
    """Write per-document completeness (how many CSV header fields are non-blank).
    The CSV header is identical to save_extraction_csv; we recompute so the
    report is always in sync.
    """
    # Keep exactly same required columns list as save_extraction_csv
    csv_header = [
        'Document_ID', 'Timestamp', 'Patient_ID', 'Patient_Created', 'Order_Pushed',
        'Patient_First_Name', 'Patient_Last_Name', 'Patient_DOB', 'Patient_Sex',
        'Medical_Record_No', 'Service_Line', 'Payer_Source', 'Physician_NPI',
        'Agency_Name', 'Patient_Address', 'Patient_City', 'Patient_State',
        'Patient_Zip', 'Patient_Phone', 'Patient_Email', 'Order_Number',
        'Order_Date', 'Start_Of_Care', 'Episode_Start_Date', 'Episode_End_Date',
        'Sent_To_Physician_Date', 'Signed_By_Physician_Date', 'Company_ID',
        'PG_Company_ID', 'SOC_Episode', 'Start_Episode', 'End_Episode',
        'Diagnosis_1', 'Diagnosis_2', 'Diagnosis_3', 'Diagnosis_4',
        'Diagnosis_5', 'Diagnosis_6'
    ]

    import csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Document_ID", "Fields_Present", "Fields_Missing", "Pct_Complete"] + csv_header)

        total_cols = len(csv_header)

        for res in results:
            # Build a dict of column→value similar to save_extraction_csv logic but simplified
            patient = res.patient_data
            order = res.order_data

            def safe(val):
                return str(val).strip() if val else ""

            col_values = {
                'Document_ID': res.doc_id,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Patient_ID': safe(patient.medical_record_no or getattr(patient, 'account_number', '')),
                'Patient_Created': '',
                'Order_Pushed': '',
                'Patient_First_Name': safe(patient.patient_fname),
                'Patient_Last_Name': safe(patient.patient_lname),
                'Patient_DOB': safe(patient.dob),
                'Patient_Sex': safe(patient.patient_sex),
                'Medical_Record_No': safe(patient.medical_record_no),
                'Service_Line': safe(getattr(order, 'service_line', '')),
                'Payer_Source': safe(patient.primary_insurance),
                'Physician_NPI': safe(patient.provider_npi),
                'Agency_Name': safe(getattr(order, 'agency_name', '')),
                'Patient_Address': safe(patient.address),
                'Patient_City': safe(patient.city),
                'Patient_State': safe(patient.state),
                'Patient_Zip': safe(patient.zip_code),
                'Patient_Phone': safe(patient.phone_number),
                'Patient_Email': safe(patient.email),
                'Order_Number': safe(order.order_no),
                'Order_Date': safe(getattr(order, 'order_date', '')),
                'Start_Of_Care': safe(order.start_of_care),
                'Episode_Start_Date': safe(order.episode_start_date),
                'Episode_End_Date': safe(order.episode_end_date),
                'Sent_To_Physician_Date': safe(getattr(order, 'sent_to_physician_date', '')),
                'Signed_By_Physician_Date': safe(getattr(order, 'signed_by_physician_date', '')),
                'Company_ID': safe(getattr(order, 'company_id', '')),
                'PG_Company_ID': safe(getattr(order, 'pg_company_id', '')),
                'SOC_Episode': safe(getattr(order, 'soc_episode', '')),
                'Start_Episode': safe(getattr(order, 'start_episode', '')),
                'End_Episode': safe(getattr(order, 'end_episode', '')),
            }
            # Diagnoses codes
            diag_codes = []
            if order.primary_diagnosis:
                diag_codes.append(order.primary_diagnosis)
            if order.secondary_diagnosis:
                diag_codes.append(order.secondary_diagnosis)
            for d in order.episode_diagnoses:
                if d.diagnosis_code:
                    diag_codes.append(d.diagnosis_code)
            while len(diag_codes) < 6:
                diag_codes.append("")
            for i in range(6):
                col_values[f'Diagnosis_{i+1}'] = safe(diag_codes[i])

            # Compute completeness
            present = sum(1 for c in csv_header if col_values.get(c, '').strip())
            missing = total_cols - present
            pct = round(present/total_cols*100, 1)

            writer.writerow([res.doc_id, present, missing, pct] + [col_values.get(c, '') for c in csv_header]) 