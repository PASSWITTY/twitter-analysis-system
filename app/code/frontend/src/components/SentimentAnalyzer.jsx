import React, { useState } from 'react';
import axios from 'axios';

const SentimentAnalyzer = () => {
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSentiment(null);
    setError(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { text });
      setSentiment(response.data.sentiment);
    } catch (err) {
      setError('An error occurred while processing your request.');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-bold mb-4">Sentiment Analyzer</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md p-8 bg-white rounded shadow-md">
        <div className="mb-4">
          <label htmlFor="text" className="block text-gray-700 text-sm font-bold mb-2">Enter text:</label>
          <textarea
            id="text"
            className="w-full px-3 py-2 text-gray-700 border rounded-lg focus:outline-none"
            rows="5"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none"
        >
          Analyze Sentiment
        </button>
      </form>
      {sentiment && (
        <div className="mt-4 p-4 bg-green-100 border-t-4 border-green-500 rounded text-green-700">
          Sentiment: {sentiment}
        </div>
      )}
      {error && (
        <div className="mt-4 p-4 bg-red-100 border-t-4 border-red-500 rounded text-red-700">
          {error}
        </div>
      )}
    </div>
  );
};

export default SentimentAnalyzer;
