"""
PDF Report Generator for Predictive Maintenance System
Generates professional PDF reports with failure predictions and analyses
"""

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
import yaml
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Professional PDF report generator for predictive maintenance"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize report generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.report_config = self.config['reports']
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.temp_dir = Path("reports/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=12,
            spaceAfter=8
        ))
    
    def _get_risk_color(self, risk_level: str) -> colors.Color:
        """Get color based on risk level"""
        color_map = {
            'CRITICAL': colors.HexColor('#e74c3c'),
            'HIGH': colors.HexColor('#e67e22'),
            'MEDIUM': colors.HexColor('#f39c12'),
            'LOW': colors.HexColor('#27ae60')
        }
        return color_map.get(risk_level, colors.grey)
    
    def _create_sensor_chart(self, sensor_data: pd.DataFrame, 
                            filename: str) -> str:
        """Create sensor readings chart"""
        plt.figure(figsize=(10, 6))
        
        # Select numeric columns
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns[:6]
        
        for col in numeric_cols:
            plt.plot(sensor_data.index, sensor_data[col], label=col, marker='o')
        
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title('Sensor Readings Trend')
        plt.legend(loc='best', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = self.temp_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_risk_gauge(self, probability: float, filename: str) -> str:
        """Create a risk gauge visualization"""
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        radius = np.ones(100)
        
        # Color segments
        colors_list = ['green', 'yellow', 'orange', 'red']
        boundaries = [0, 0.25, 0.5, 0.75, 1.0]
        
        for i in range(4):
            mask = (theta >= boundaries[i] * np.pi) & (theta <= boundaries[i+1] * np.pi)
            ax.fill_between(theta[mask], 0, radius[mask], 
                           color=colors_list[i], alpha=0.7)
        
        # Add needle
        needle_angle = (probability / 100) * np.pi
        ax.plot([needle_angle, needle_angle], [0, 0.9], 
               color='black', linewidth=3)
        ax.scatter([needle_angle], [0.9], color='black', s=100, zorder=5)
        
        ax.set_ylim(0, 1)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_title(f'Failure Probability: {probability:.1f}%', pad=20)
        
        plt.tight_layout()
        filepath = self.temp_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def generate_failure_report(self,
                               machine_id: str,
                               prediction_result: Dict,
                               sensor_data: pd.DataFrame = None,
                               historical_data: pd.DataFrame = None,
                               output_filename: str = None) -> str:
        """
        Generate comprehensive failure prediction PDF report
        
        Args:
            machine_id: Machine identifier
            prediction_result: Prediction results from FailurePredictor
            sensor_data: Current sensor readings
            historical_data: Historical sensor data (optional)
            output_filename: Custom output filename
            
        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating failure report for {machine_id}...")
        
        # Setup output file
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"failure_report_{machine_id}_{timestamp}.pdf"
        
        output_dir = Path(self.report_config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        # Create PDF
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        
        # Extract prediction data
        probability = prediction_result.get('failure_probability', 0)
        risk_level = prediction_result.get('risk_level', 'UNKNOWN')
        confidence = prediction_result.get('confidence', 0)
        days_to_failure = prediction_result.get('estimated_days_to_failure', 'N/A')
        root_causes = prediction_result.get('root_causes', [])
        recommendation = prediction_result.get('maintenance_recommendation', 'N/A')
        
        # ===== TITLE =====
        title = Paragraph(
            f"Predictive Maintenance Report<br/>Machine ID: {machine_id}",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 20))
        
        # ===== EXECUTIVE SUMMARY =====
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        summary_data = [
            ['Report Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Machine ID:', machine_id],
            ['Failure Probability:', f"{probability:.1f}%"],
            ['Risk Level:', risk_level],
            ['Confidence:', f"{confidence*100:.1f}%"],
            ['Estimated Time to Failure:', f"{days_to_failure} days"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # ===== RISK GAUGE =====
        gauge_file = self._create_risk_gauge(probability, f"gauge_{machine_id}.png")
        gauge_img = RLImage(gauge_file, width=4*inch, height=2.5*inch)
        story.append(gauge_img)
        story.append(Spacer(1, 20))
        
        # ===== ROOT CAUSE ANALYSIS =====
        story.append(Paragraph("Root Cause Analysis", self.styles['SectionHeader']))
        story.append(Paragraph(
            "The following sensors/parameters show the highest contribution to the failure prediction:",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 10))
        
        if root_causes:
            cause_data = [['Rank', 'Parameter', 'Current Value', 'Importance Score']]
            for i, cause in enumerate(root_causes[:5], 1):
                cause_data.append([
                    str(i),
                    cause.get('feature', 'N/A'),
                    f"{cause.get('value', 0):.2f}",
                    f"{cause.get('importance', 0):.4f}"
                ])
            
            cause_table = Table(cause_data, colWidths=[0.6*inch, 2.5*inch, 1.5*inch, 1.5*inch])
            cause_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
            ]))
            story.append(cause_table)
        else:
            story.append(Paragraph("No root cause data available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # ===== MAINTENANCE RECOMMENDATIONS =====
        story.append(Paragraph("Maintenance Recommendations", self.styles['SectionHeader']))
        
        # Color code based on risk
        risk_color = self._get_risk_color(risk_level)
        rec_style = ParagraphStyle(
            'Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=16,
            leftIndent=20
        )
        
        story.append(Paragraph(
            f'<font color="{risk_color.hexval()}"><b>Risk Level: {risk_level}</b></font>',
            self.styles['Normal']
        ))
        story.append(Spacer(1, 10))
        
        for line in recommendation.split('\n'):
            if line.strip():
                story.append(Paragraph(line, rec_style))
        
        story.append(Spacer(1, 20))
        
        # ===== SENSOR DATA (if provided) =====
        if sensor_data is not None and len(sensor_data) > 0:
            story.append(PageBreak())
            story.append(Paragraph("Technical Analysis - Sensor Readings", 
                                  self.styles['SectionHeader']))
            
            # Create sensor chart
            chart_file = self._create_sensor_chart(sensor_data, f"sensors_{machine_id}.png")
            chart_img = RLImage(chart_file, width=6*inch, height=3.5*inch)
            story.append(chart_img)
            story.append(Spacer(1, 20))
            
            # Sensor data table (show first few rows)
            story.append(Paragraph("Current Sensor Values", self.styles['SubHeader']))
            numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns[:8]
            
            if len(numeric_cols) > 0:
                sensor_table_data = [['Parameter', 'Value']]
                for col in numeric_cols:
                    value = sensor_data[col].iloc[-1] if len(sensor_data) > 0 else 0
                    sensor_table_data.append([col, f"{value:.2f}"])
                
                sensor_table = Table(sensor_table_data, colWidths=[3*inch, 2*inch])
                sensor_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
                ]))
                story.append(sensor_table)
        
        # ===== FOOTER =====
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "<i>This report is generated by an AI-based predictive maintenance system. "
            "Always consult with maintenance professionals for critical decisions.</i>",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<i>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</i>",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✓ Report generated: {output_path}")
        return str(output_path)


if __name__ == "__main__":
    # Test report generator
    generator = ReportGenerator()
    logger.info("ReportGenerator initialized successfully")
