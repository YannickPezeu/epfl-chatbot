import { NextResponse } from 'next/server';

const BASE_URL_LOCAL = process.env.BASE_URL_LOCAL || 'http://localhost:8000';
const BASE_URL_ONLINE_TEST = process.env.BASE_URL_ONLINE_TEST || 'http://lex-chatbot-backend-service-test3:8000';
const BASE_URL_ONLINE = process.env.BASE_URL_ONLINE || 'http://lex-chatbot-backend-service3:8000';

function getBaseUrl(headers) {
  const host = headers.get('host') || '';
  
if (host.includes('test')) {
    return BASE_URL_ONLINE_TEST;
  } else {
    return BASE_URL_ONLINE;
  }
}

export async function GET(request) {
  const baseUrl = getBaseUrl(request.headers);

  console.log('baseUrl:', baseUrl);

  try {
    // Make a request to your backend service
    const response = await fetch(`${baseUrl}/your-endpoint`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        // Add any other headers your backend requires
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Return the data from your backend
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in /myapi route:', error);
    return NextResponse.json({ error: 'Internal Server Error', details: error.message }, { status: 500 });
  }
}