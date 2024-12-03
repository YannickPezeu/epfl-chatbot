import Link from 'next/link'
import styles from './styles/Home.module.css'

export default function Home() {
  return (
    <div className={styles.container}>
      <h1>Welcome to the legal ChatBot</h1>
      <div className={styles.cardContainer}>
        {/* <Link href="/RH" className={styles.card}>
          <h2>Human Resources</h2>
          <p>Chatbot able to explore the content of the HR website and the HR LEXs</p>
        </Link>
        <Link href="/LEX" className={styles.card}>
          <h2>LEXs</h2>
          <p>Chatbot able to explore the content of all EPFL LEXs</p>
        </Link>
        <Link href="/test" className={styles.card}>
          <h2>Test</h2>
          <p>A space for tools in development</p>

        </Link> */}
        <Link href="/RH_chatbot_prototype" className={styles.card}>
          <h2>Human Ressources chatbot</h2>
          <p>A chatbot with libraries containing document related to human ressources. </p>
        </Link>
        {/* <Link href="/email_handler" className={styles.card}>
          <h2>email_handler</h2>
          <p>A chatbot handling email with classification in 3 categories: danger, out of scope, in the scope. </p>
        </Link> */}
      </div>
    </div>
  )
}