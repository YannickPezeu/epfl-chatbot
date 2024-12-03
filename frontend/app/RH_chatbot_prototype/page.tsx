'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import styles from './Personal.module.css';
import SignupModal from '../components/SignupModal/SignupModal';
import Main from '../components/Main/Main';
import {useStore} from '../store';


export default function Personal() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [shouldTryLogin, setShouldTryLogin] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const [fullURL, setFullURL] = useState('');


  const router = useRouter();
  const [isBaseUrlSet, setIsBaseUrlSet] = useState(false);
  const [hasCheckedLoginStatus, setHasCheckedLoginStatus] = useState(false);


  const {
    setBASE_URL,
    setBASE_URL_WS,
    BASE_URL,
    BASE_URL_LOCAL,
    BASE_URL_WS_LOCAL,
    BASE_URL_ONLINE,
    BASE_URL_ONLINE_WS,
    BASE_URL_ONLINE_TEST,
    BASE_URL_ONLINE_WS_TEST,
    isLoggedIn,
    setIsLoggedIn,
    interaction_type,
    setInteractionType,

  } = useStore();
  if(interaction_type !== 'chat'){ setInteractionType('chat');}
  useEffect(() => {
    const fullURL = window.location.href;
    if(fullURL.includes('test')) {
      setBASE_URL(BASE_URL_ONLINE_TEST);
      setBASE_URL_WS(BASE_URL_ONLINE_WS_TEST);
    } else{
      setBASE_URL(BASE_URL_ONLINE);
      setBASE_URL_WS(BASE_URL_ONLINE_WS);
    }
    setIsBaseUrlSet(true);
  }, []);


  



  const handleSignup = async (newUsername: string, newPassword: string) => {
    try {
      const urlWithParameter = `${BASE_URL}/auth/signup?username=${encodeURIComponent(newUsername)}&password=${encodeURIComponent(newPassword)}`;
      const response = await fetch(urlWithParameter, {
        method: 'POST',
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Set the username and password, then trigger login
          setUsername(newUsername);
          setPassword(newPassword);
          setShouldTryLogin(true);
          // Create a mock event for handleLogin
          const mockEvent = {
            preventDefault: () => {}
        } as React.FormEvent;
        
        // Call handleLogin
        await handleLogin(mockEvent);
        } else {
          setError(data.message || 'Signup failed');
        }
      } else {
        setError('Signup failed. Please try again.');
      }
    } catch (error) {
      console.error('Error during signup:', error);
      setError('An error occurred during signup');
    }
  };


  const checkLoginStatus = useCallback(async () => {
    if (hasCheckedLoginStatus || BASE_URL === "http://") return;

    try {
      console.log("Checking login status with BASE_URL:", BASE_URL);
      const response = await fetch(`${BASE_URL}/auth/check_session`, {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data = await response.json();
        setIsLoggedIn(data.is_logged_in);
        if (data.is_logged_in) {
          setUsername(data.username);
        }
      }
    } catch (error) {
      console.error('Error checking login status:', error);
    } finally {
      setHasCheckedLoginStatus(true);
    }
  }, [BASE_URL, hasCheckedLoginStatus, setIsLoggedIn]);

  useEffect(() => {
    checkLoginStatus();
  }, [checkLoginStatus]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
        const urlWithParameter = `${BASE_URL}/auth/login?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`;
        const response = await fetch(urlWithParameter, {
            method: 'POST',
            credentials: 'include',
        });
        
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
              setIsLoggedIn(true);
              setError('');
          } else {
                setError(data.message || 'Login failed');
            }
        } else {
            setError('Invalid username or password');
        }
    } catch (error) {
        console.error('Error during login:', error);
        setError('An error occurred during login');
    }
};

  if (isLoggedIn) {
    return (
      <Main
      extend={false}
      ></Main>
    );
  }

  return (
    <div className={styles.container}>
      <h1>Login to Personal Page</h1>
      <form onSubmit={handleLogin} className={styles.form}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className={styles.input}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className={styles.input}
        />
        <button type="submit" className={styles.button}>Login</button>
      </form>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.signupContainer}>
        <p>Don&apos;t have an account?</p>
        <button onClick={() => setIsSignupModalOpen(true)} className={styles.signupButton}>Sign Up</button>
      </div>
      <SignupModal
        isOpen={isSignupModalOpen}
        onClose={() => setIsSignupModalOpen(false)}
        onSignup={handleSignup}
      />
    </div>
  );
}

