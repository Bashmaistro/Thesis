import React, { useState } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Legend,
  ResponsiveContainer,
} from 'recharts';



const sourceBreakdown = [
  { name: 'Clinic', value: 2 },
  { name: 'MRI', value: 33 },
  { name: 'Histopathology', value: 66 },
];

const COLORS = ['#8884d8', '#82ca9d', '#61dafb'];

const images = [
  '/images/histo1.png',
  '/images/histo2.png',
  '/images/histo3.png',
  '/images/histo4.png',
  '/images/histo5.png',
  '/images/histo6.png',
  '/images/histo7.png',
  '/images/histo8.png',
  '/images/histo9.png',
  '/images/histo10.png',
];

const progressData = [
  { day: '0-125', percentage: 10 },
  { day: '126-325', percentage: 55 },
  { day: '326-625', percentage: 15 },
  { day: '626-1825', percentage: 10 },
  { day: '1825+', percentage: 10 },
];

const tumorType = 'Astrocytoma';
const grade = 'Grade II';
const patientId = '128348203';

export default function HistopathologyDashboard() {
  const [hoveredImage, setHoveredImage] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });

  const [showMRIHover, setShowMRIHover] = useState(false);
  const [mriHoverPosition, setMriHoverPosition] = useState({ x: 0, y: 0, xPercent: 50, yPercent: 50 });


  const handleMRIMouseMove = (e) => {
  setShowMRIHover(true);
  const bounds = e.currentTarget.getBoundingClientRect();

  const offsetX = e.clientX - bounds.left;
  const offsetY = e.clientY - bounds.top;

  const xPercent = (offsetX / bounds.width) * 100;
  const yPercent = (offsetY / bounds.height) * 100;

  setMriHoverPosition({ x: e.clientX - 120, y: e.clientY + 20, xPercent, yPercent });
};

  const handleMRIMouseLeave = () => {
    setShowMRIHover(false);
  };

  const handleMouseEnter = (index) => {
    setHoveredImage(index);
  };

  const handleMouseMove = (e) => {
    const bounds = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - bounds.left) / bounds.width) * 100;
    const y = ((e.clientY - bounds.top) / bounds.height) * 100;
    setHoverPosition({ x, y });
  };

  const handleMouseLeave = () => {
    setHoveredImage(null);
  };

  const mriImage = '/images/1.png';

  return (
    <div style={{ padding: '20px', paddingTop: '40px', backgroundColor: '#282c34', minHeight: '100vh', color: 'rgb(40, 44, 52)' }}>
      
                {/* Header */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  backgroundColor: '#1f232a',
                  padding: '16px 24px',
                  borderRadius: '16px',
                  marginBottom: '24px',
                  marginTop: '0px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.4)'
                }}>
                  {/* Left Side - Navigation */}
                  <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#61dafb' }}>🆔 Patient ID: <span style={{ color: '#fff' }}>{patientId}</span></div>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: '#bbb', cursor: 'pointer', transition: 'color 0.3s' }}
                      onMouseOver={e => e.target.style.color = '#61dafb'}
                      onMouseOut={e => e.target.style.color = '#bbb'}
                    >
                      Predict
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: '#bbb', cursor: 'pointer', transition: 'color 0.3s' }}
                      onMouseOver={e => e.target.style.color = '#61dafb'}
                      onMouseOut={e => e.target.style.color = '#bbb'}
                    >
                      Patients
                    </div>
                  </div>

                  {/* Right Side - Patient ID */}
                  <div style={{ fontSize: '14px', color: '#999', fontWeight: '500' }}>
                    🧠 MedPredict
                  </div>
                </div>

      <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: '#61dafb' }}>
        Histopathology Dashboard
      </h1>

      <div style={{ display: 'flex', gap: '24px' }}>
        {/* Histopathology Images */}
        <div style={{ flex: 3, display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
          {images.map((src, index) => (
            <div
              key={index}
              style={{
                overflow: 'hidden',
                borderRadius: '16px',
                backgroundColor: '#3a3f47',
                boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.5)',
                cursor: 'pointer',
              }}
              onMouseEnter={() => handleMouseEnter(index)}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              <img
                src={src}
                alt={`Histopathology ${index + 1}`}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  transition: 'transform 0.3s ease',
                  transform: hoveredImage === index ? 'scale(2.5)' : 'scale(1)',
                  transformOrigin:
                    hoveredImage === index
                      ? `${hoverPosition.x}% ${hoverPosition.y}%`
                      : 'center',
                }}
              />
            </div>
          ))}
        </div>

        {/* MRI Image */}
        <div
          style={{
            flex: 1,
            backgroundColor: '#3a3f47',
            borderRadius: '16px',
            padding: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
          }}
          onMouseMove={handleMRIMouseMove}
          onMouseLeave={handleMRIMouseLeave}
        >
          <img
            src="/images/1.png"
            alt="MRI"
            style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '12px' }}
          />
        </div>

        {/* MRI Hover Zoom */}
        {showMRIHover && (
            <div
              style={{
                position: 'fixed',
                top: `${mriHoverPosition.y}px`,
                left: `${mriHoverPosition.x}px`,
                width: '350px',
                height: '350px',
                border: '2px solid #61dafb',
                borderRadius: '12px',
                overflow: 'hidden',
                zIndex: 9999,
                backgroundImage: `url('/images/2.png')`,
                backgroundSize: '400%', // Zoom seviyesi (artırmak için 300% de yapabilirsin)
                backgroundPosition: `${mriHoverPosition.xPercent}% ${mriHoverPosition.yPercent}%`,
                boxShadow: '0 0 10px rgba(0,0,0,0.5)',
                pointerEvents: 'none',
              }}
            />
          )}
      </div>


      {/* Prediction + Pie Chart */}
      <div style={{ display: 'flex', gap: '24px', marginTop: '24px' }}>
        {/* Life Expectancy */}
        <div style={{ flex: 1, backgroundColor: '#3a3f47', borderRadius: '50px', padding: '16px', boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.5)' }}>
          <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#61dafb' }}>
            Life Expectancy Predictions
          </h2>
          <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '8px' }}>
            {tumorType} - {grade}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {progressData.map((item, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '60px', color: '#61dafb' }}>Day {item.day}</div>
                <div style={{ flex: 1, height: '8px', backgroundColor: '#555', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${item.percentage}%`, height: '100%', backgroundColor: '#61dafb' }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>

                 {/* Pie Chart */}
          {/* Pie Chart */}
            <div
              style={{
                flex: 1,
                maxWidth: '400px', // Max genişlik kısıtlaması, gerektiğinde kaldırabilirsin
                aspectRatio: '1', // Kare oran
                backgroundColor: '#3a3f47',
                borderRadius: '16px',
                padding: '12px',
                boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.4)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
              }}
            >
              <h2
                style={{
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#61dafb',
                  textAlign: 'center',
                  marginBottom: '8px',
                }}
              >
                Attention distribution across modalities
              </h2>
              <div style={{ width: '100%', height: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sourceBreakdown}
                      cx="50%"
                      cy="50%"
                      outerRadius="80%"
                      label={({ name, percent }) =>
                        `${name}: ${(percent * 100).toFixed(0)}%`
                      }
                      dataKey="value"
                    >
                      {sourceBreakdown.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Legend verticalAlign="bottom" iconSize={10} height={36} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
      </div>
    </div>
  );
} 