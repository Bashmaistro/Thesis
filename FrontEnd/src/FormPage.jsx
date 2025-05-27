import React, { useState } from 'react';

const UploadForm = ({ onResult }) => {
  const [mriFile, setMriFile] = useState(null);
  const [histologyFile, setHistologyFile] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [isLoading, setIsLoading] = useState(false); // 👈 loading state

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!mriFile || !histologyFile) {
      alert("Lütfen tüm dosyaları seçin.");
      return;
    }

    const formData = new FormData();
    formData.append('mri', mriFile);
    formData.append('histo', histologyFile);
    formData.append('age', age);
    formData.append('gender', gender);

    setIsLoading(true); // 👈 yükleme başladı

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      setIsLoading(false); // 👈 yükleme bitti

      if (result.status === 'success') {
        onResult(result);
      } else {
        alert('Sunucuda hata oluştu.');
      }
    } catch (error) {
      setIsLoading(false); // 👈 hata anında da bitmeli
      alert('Yükleme başarısız.');
      console.error(error);
    }
  };

  return (
    <div style={styles.container}>
      {isLoading ? (
        <div style={styles.loadingContainer}>
          <div className="spinner" style={styles.spinner}></div>
          <p>Veriler işleniyor, lütfen bekleyin...</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} style={styles.form}>
          <h2 style={styles.header}>Upload Patient Data</h2>

          <label style={styles.label}>MRI File (.nii/.nii.gz)</label>
          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={e => setMriFile(e.target.files[0])}
            required
            style={styles.input}
          />

          <label style={styles.label}>Histopathology Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={e => setHistologyFile(e.target.files[0])}
            required
            style={styles.input}
          />

          <label style={styles.label}>Age</label>
          <input
            type="number"
            value={age}
            onChange={e => setAge(e.target.value)}
            required
            style={styles.input}
          />

          <label style={styles.label}>Gender</label>
          <select
            value={gender}
            onChange={e => setGender(e.target.value)}
            required
            style={styles.input}
          >
            <option value="">Select Gender</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>

          <button type="submit" style={styles.button}>Submit</button>
        </form>
      )}
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    backgroundColor: '#f5f5f5',
  },
  form: {
    backgroundColor: '#fff',
    padding: '40px',
    borderRadius: '8px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    width: '100%',
    maxWidth: '500px',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    marginBottom: '20px',
    fontSize: '24px',
    textAlign: 'center',
  },
  label: {
    marginTop: '15px',
    marginBottom: '5px',
    fontSize: '14px',
  },
  input: {
    padding: '10px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    fontSize: '14px',
  },
  button: {
    marginTop: '20px',
    padding: '12px',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
  },
  loadingContainer: {
    textAlign: 'center',
    fontSize: '18px',
  },
  spinner: {
    margin: '0 auto 20px',
    border: '6px solid #f3f3f3',
    borderTop: '6px solid #007bff',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    animation: 'spin 1s linear infinite',
  }
};

// Spinner animasyonu eklemek için CSS gerekiyor:
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(styleSheet);

export default UploadForm;
