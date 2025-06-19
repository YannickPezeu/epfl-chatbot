// epfl-finance-kb-scraper.js
const puppeteer = require('puppeteer');
const fs = require('fs-extra');
const path = require('path');
const dotenv = require('dotenv');
const { setTimeout } = require('timers/promises');

// Charger le fichier .env depuis le répertoire courant
const envPath = path.resolve(__dirname, '../.env');
dotenv.config({ path: envPath });

// Vérifier que les identifiants sont disponibles
if (!process.env.EPFL_USERNAME || !process.env.EPFL_PASSWORD) {
  console.error('Erreur: Les identifiants EPFL_USERNAME et EPFL_PASSWORD doivent être définis dans le fichier .env');
  process.exit(1);
}

// Configuration
const config = {
  startUrl: 'https://support.epfl.ch/epfl?id=epfl_kb_home',
  outputDir: path.join(__dirname, 'epfl_finance_kb'),
  credentials: {
    username: process.env.EPFL_USERNAME,
    password: process.env.EPFL_PASSWORD
  },
  // Délais pour les actions (en ms)
  delays: {
    navigation: 5000,      // Attente après navigation
    scroll: 1000,          // Délai entre les scrolls
    typing: 100,           // Délai entre les frappes
    download: 3000,        // Attente pour téléchargement
    randomMin: 500,        // Délai aléatoire minimum
    randomMax: 2000        // Délai aléatoire maximum
  },
  // Rotation des user agents
  userAgents: [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
  ]
};

// Fonction pour obtenir un délai aléatoire pour paraître plus humain
function getRandomDelay() {
  return Math.floor(Math.random() * (config.delays.randomMax - config.delays.randomMin + 1)) + config.delays.randomMin;
}

// Fonction pour obtenir un User-Agent aléatoire
function getRandomUserAgent() {
  return config.userAgents[Math.floor(Math.random() * config.userAgents.length)];
}

// Fonction pour attendre avec un délai aléatoire
async function waitRandomTime(baseTime = 0) {
  const randomTime = getRandomDelay();
  await setTimeout(baseTime + randomTime);
}

// Fonction principale
async function main() {
  console.log('Démarrage du scraper de la base de connaissances financière EPFL...');
  console.log(`Utilisation du compte: ${config.credentials.username}`);

  // Créer le dossier de sortie s'il n'existe pas
  fs.ensureDirSync(config.outputDir);

  // Fichier de log pour suivi de progression
  const logFile = path.join(config.outputDir, 'log.txt');
  fs.appendFileSync(logFile, `[${new Date().toISOString()}] Démarrage du scraper\n`);

  // Liste des articles déjà téléchargés (pour reprendre en cas d'interruption)
  const downloadedListFile = path.join(config.outputDir, 'downloaded.json');
  let downloadedArticles = [];

  if (fs.existsSync(downloadedListFile)) {
    try {
      downloadedArticles = JSON.parse(fs.readFileSync(downloadedListFile, 'utf8'));
      console.log(`${downloadedArticles.length} articles déjà téléchargés trouvés.`);
    } catch (error) {
      console.error(`Erreur lors de la lecture de la liste des articles téléchargés: ${error.message}`);
    }
  }

  // Lancer le navigateur
  const browser = await puppeteer.launch({
    headless: false, // false pour pouvoir voir ce qui se passe (mettre à true en production)
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--window-size=1920x1080',
    ],
    defaultViewport: { width: 1920, height: 1080 }
  });

  try {
    // Ouvrir une nouvelle page
    const page = await browser.newPage();

    // Configurer le User-Agent
    await page.setUserAgent(getRandomUserAgent());

    // Augmenter les timeouts
    await page.setDefaultNavigationTimeout(60000);
    await page.setDefaultTimeout(30000);

    // Activer la possibilité de télécharger des PDF
    const client = await page.target().createCDPSession();
    await client.send('Page.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: config.outputDir
    });

    // Étape 1: Accéder à la page d'accueil
    console.log('Accès à la page d\'accueil...');
    await page.goto(config.startUrl, { waitUntil: 'networkidle2' });
    await waitRandomTime(config.delays.navigation);

    // Étape 2: Cliquer sur le bouton "Log in"
    console.log('Clic sur le bouton "Log in"...');
    await page.click('.nav.navbar-nav.login a.pulseit');
    await waitRandomTime(config.delays.navigation);

    // Étape 3: Attendre et gérer la page de connexion Microsoft
    console.log('Attente de la page de connexion Microsoft...');

    // Attendre que la page de connexion soit chargée (URL contenant microsoftonline.com)
    await page.waitForFunction(
      'window.location.href.includes("microsoftonline.com")',
      { timeout: 30000 }
    );
    await waitRandomTime();

    // Étape 4: Remplir le champ email
    console.log('Remplissage du champ email...');
    await page.waitForSelector('#i0116');

    // Taper le nom d'utilisateur lettre par lettre (plus humain)
    for (const char of config.credentials.username) {
      await page.type('#i0116', char);
      await setTimeout(config.delays.typing);
    }

    await waitRandomTime();

    // Étape 5: Cliquer sur "Next"
    console.log('Clic sur "Next"...');
    await page.click('#idSIButton9');
    await waitRandomTime(config.delays.navigation);

    // Étape 6: Remplir le champ mot de passe
    console.log('Remplissage du mot de passe...');
    await page.waitForSelector('#i0118');

    // Taper le mot de passe lettre par lettre (plus humain)
    for (const char of config.credentials.password) {
      await page.type('#i0118', char);
      await setTimeout(config.delays.typing);
    }

    await waitRandomTime();

    // Étape 7: Cliquer sur "Sign in"
    console.log('Clic sur "Sign in"...');
    await page.click('#idSIButton9');
    await waitRandomTime(config.delays.navigation);

    // Si un écran "Rester connecté" apparaît, cliquer sur "Oui"
    try {
      await page.waitForSelector('#idSIButton9', { timeout: 5000 });
      console.log('Écran "Rester connecté" détecté, clic sur "Oui"...');
      await page.click('#idSIButton9');
      await waitRandomTime(config.delays.navigation);
    } catch (error) {
      console.log('Pas d\'écran "Rester connecté"');
    }

    // Attendre que la redirection soit terminée et qu'on soit de retour sur le portail EPFL
    console.log('Attente de la redirection vers le portail EPFL...');
    await page.waitForFunction(
      'window.location.href.includes("support.epfl.ch")',
      { timeout: 60000 }
    );
    await waitRandomTime(config.delays.navigation);

    // Étape 8: Cliquer sur "Base de connaissances financière"
    console.log('Clic sur "Base de connaissances financière"...');

    // Attendre que la page soit entièrement chargée
    await page.waitForSelector('.kb-center', { visible: true });

    // Trouver le bon lien (celui qui contient le texte "Base de connaissances financière")
    const financialKBSelector = '.kb-center h5.ng-binding';
    await page.waitForFunction(
      `Array.from(document.querySelectorAll('${financialKBSelector}')).some(el => el.innerText.includes('Base de connaissances financière'))`,
      { timeout: 10000 }
    );

    // Cliquer sur l'élément correspondant
    await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('.kb-center h5.ng-binding'));
      const targetElement = elements.find(el => el.innerText.includes('Base de connaissances financière'));
      if (targetElement) {
        targetElement.closest('.kb-center').click();
      }
    });

    await waitRandomTime(config.delays.navigation);

    // Étape 9: Attendre le chargement de la liste des articles
    console.log('Attente du chargement de la liste des articles...');
    await page.waitForSelector('.kb-article-summary', { visible: true });
    await waitRandomTime();

    // Étape 10: Scroller pour charger tous les articles (environ 110)
    console.log('Chargement de tous les articles en scrollant...');

    let previousArticleCount = 0;
    let currentArticleCount = 0;
    let noChangeCount = 0;
    let maxScrollAttempts = 20;
    let scrollAttempt = 0;

    // Scroller jusqu'à ce qu'on ait tous les articles ou qu'il n'y ait plus de changement
    do {
      previousArticleCount = currentArticleCount;

      // Utiliser une méthode de scrolling plus efficace avec une approche progressive
      await page.evaluate(() => {
        // Scroller par incréments pour simuler un comportement plus humain
        const totalHeight = document.body.scrollHeight;
        const viewportHeight = window.innerHeight;
        const scrollStep = viewportHeight / 2; // Scroller par demi-écran

        let currentPosition = window.scrollY;
        const targetPosition = Math.min(currentPosition + scrollStep, totalHeight - viewportHeight);

        // Animation de défilement fluide
        const scrollToSmoothly = (target, duration) => {
          const start = window.scrollY;
          const change = target - start;
          const startTime = performance.now();

          const animateScroll = (currentTime) => {
            const elapsedTime = currentTime - startTime;
            if (elapsedTime > duration) {
              window.scrollTo(0, target);
              return;
            }

            const progress = elapsedTime / duration;
            const easeInOutCubic = progress < 0.5
              ? 4 * progress * progress * progress
              : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            window.scrollTo(0, start + change * easeInOutCubic);
            requestAnimationFrame(animateScroll);
          };

          requestAnimationFrame(animateScroll);
        };

        // Exécuter l'animation de défilement sur 300ms
        scrollToSmoothly(targetPosition, 300);

        // Assurer que nous atteignons bien le bas de la page après plusieurs tentatives
        if (targetPosition >= totalHeight - viewportHeight - 10) {
          // Forcer un scroll complet jusqu'au bas si on est presque en bas
          window.scrollTo(0, totalHeight);
        }

        return {
          currentPosition,
          targetPosition,
          totalHeight,
          viewportHeight,
          isAtBottom: (targetPosition >= totalHeight - viewportHeight - 10)
        };
      });

      // Attendre que le contenu se charge après le scroll
      await waitRandomTime(config.delays.scroll * 1.5);

      // Compter le nombre d'articles actuellement chargés
      currentArticleCount = await page.evaluate(() => {
        return document.querySelectorAll('.kb-article-summary').length;
      });

      scrollAttempt++;
      console.log(`Articles chargés: ${currentArticleCount} (tentative ${scrollAttempt}/${maxScrollAttempts})`);

      // Si le nombre d'articles ne change pas après plusieurs tentatives, on continue quand même quelques fois
      if (currentArticleCount === previousArticleCount) {
        noChangeCount++;

        // Si on a plusieurs tentatives sans changement, on essaie un scroll forcé jusqu'en bas
        if (noChangeCount >= 3) {
          await page.evaluate(() => {
            window.scrollTo(0, document.body.scrollHeight);
          });
          await waitRandomTime(config.delays.scroll * 2);
        }
      } else {
        noChangeCount = 0;
      }

      // Afficher un message pour indiquer à l'utilisateur qu'il peut scroller manuellement si nécessaire
      if (scrollAttempt === 10) {
        console.log('NOTE: Si tous les articles ne se chargent pas automatiquement, vous pouvez scroller manuellement dans le navigateur.');
      }

    } while (noChangeCount < 5 && scrollAttempt < maxScrollAttempts); // Arrêter après 5 tentatives sans nouveaux articles ou max 20 tentatives

    // Si l'utilisateur scrolle manuellement, on doit attendre que le nombre d'articles se stabilise
    console.log('Attente de 15 secondes pour permettre un scrolling manuel si nécessaire...');
    await waitRandomTime(15000);

    // Vérifier une dernière fois le nombre d'articles
    currentArticleCount = await page.evaluate(() => {
      return document.querySelectorAll('.kb-article-summary').length;
    });

    console.log(`Total des articles trouvés: ${currentArticleCount}`);
    if (currentArticleCount < 100) {
      console.log('ATTENTION: Moins de 100 articles trouvés. Il est possible que tous les articles ne soient pas chargés.');
      console.log('Vous pouvez continuer avec ce nombre ou arrêter le script et réessayer avec un scrolling manuel.');
      // Pause pour donner le temps à l'utilisateur de décider
      await waitRandomTime(10000);
    }

    console.log(`Total des articles trouvés: ${currentArticleCount}`);
    fs.appendFileSync(logFile, `[${new Date().toISOString()}] ${currentArticleCount} articles trouvés\n`);

    // Étape 11: Extraire les informations de base sur tous les articles
    console.log('Extraction des informations des articles...');

    const articles = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('.kb-article-summary')).map(article => {
        const titleElement = article.querySelector('.kb-title');
        const id = titleElement ? titleElement.href.match(/sys_kb_id=([^&]+)/)?.[1] || null : null;
        const title = titleElement ? titleElement.innerText.trim() : 'Sans titre';
        const views = article.querySelector('.kb-detail span[title="Nombre de vues"]')?.innerText.trim() || 'Inconnu';
        const lastUpdate = article.querySelector('sn-time-ago time')?.title || 'Inconnue';

        return { id, title, views, lastUpdate, url: titleElement ? titleElement.href : null };
      });
    });

    // Sauvegarder la liste des articles
    fs.writeJSONSync(
      path.join(config.outputDir, 'articles.json'),
      articles,
      { spaces: 2 }
    );

    console.log(`${articles.length} articles analysés et sauvegardés dans articles.json`);

    // Étape 12: Parcourir chaque article et le télécharger en PDF
    console.log('Début du téléchargement des articles...');

    // Créer une page séparée pour le téléchargement des articles
    const articlePage = await browser.newPage();
    await articlePage.setUserAgent(getRandomUserAgent());
    await articlePage.setDefaultNavigationTimeout(60000);
    await articlePage.setDefaultTimeout(30000);

    // Optimiser les PDFs
    await articlePage.emulateMediaType('screen');

    for (let i = 0; i < articles.length; i++) {
      const article = articles[i];

      // // Vérifier si l'article a déjà été téléchargé
      // if (downloadedArticles.some(a => a.id === article.id)) {
      //   console.log(`Article déjà téléchargé: ${article.title} (${i+1}/${articles.length})`);
      //   continue;
      // }

      console.log(`Traitement de l'article: ${article.title} (${i+1}/${articles.length})`);

      try {
        // Nettoyer le titre pour le nom de fichier
        const safeTitle = article.title.replace(/[\/\\:*?"<>|]/g, '_').substring(0, 100);
        const filename = `${article.id}_${safeTitle}.pdf`;
        const filePath = path.join(config.outputDir, filename);

        // Si l'URL n'est pas valide, ignorer cet article
        if (!article.url) {
          console.error(`URL manquante pour l'article: ${article.title}`);
          continue;
        }

        // Accéder à la page de l'article
        await articlePage.goto(article.url, { waitUntil: 'networkidle2' });
        await waitRandomTime(config.delays.navigation);

        // Attendre que le contenu de l'article soit chargé
        const contentSelector = '.v619f923b4f33df00fe35adee0310c771.ng-scope';
        try {
          await articlePage.waitForSelector(contentSelector, { timeout: 10000 });
        } catch (error) {
          console.log(`Sélecteur "${contentSelector}" non trouvé, tentative avec contenu générique...`);
          await articlePage.waitForSelector('article', { timeout: 5000 });
        }

        // Attendre un peu plus pour que tout le contenu soit bien chargé
        await waitRandomTime();

        // Générer le PDF en n'imprimant que le contenu de l'article
        console.log(`Génération du PDF pour: ${article.title}`);

        // Identifier tous les sélecteurs possibles de contenus d'articles
        const possibleSelectors = [
          '.v619f923b4f33df00fe35adee0310c771.ng-scope',
          '.kb-article-content',
          'article .kb-article-content',
          '.article-content'
        ];

        // Modifier la page pour cacher tout sauf le contenu de l'article et ajouter une bordure pour contrôle visuel
        await articlePage.evaluate((selectors) => {
          let articleElement = null;

          // Chercher l'élément du contenu de l'article parmi les sélecteurs possibles
          for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
              articleElement = element;
              break;
            }
          }

          if (articleElement) {
            // Ajouter le titre de l'article au début du contenu
            const title = document.querySelector('.kb-title')?.innerText ||
                        document.querySelector('h1')?.innerText ||
                        'Article sans titre';

            // Créer un nouvel élément div qui contiendra uniquement le contenu
            const container = document.createElement('div');
            container.id = 'article-content-for-pdf';
            container.style.padding = '20px';
            container.style.fontFamily = 'Arial, sans-serif';

            // Ajouter le titre
            const titleElement = document.createElement('h1');
            titleElement.innerText = title;
            titleElement.style.fontSize = '24px';
            titleElement.style.marginBottom = '20px';
            container.appendChild(titleElement);

            // Ajouter une ligne horizontale après le titre
            const hr = document.createElement('hr');
            hr.style.marginBottom = '20px';
            container.appendChild(hr);

            // Copier le contenu de l'article
            container.appendChild(articleElement.cloneNode(true));

            // Remplacer tout le body par notre container
            document.body.innerHTML = '';
            document.body.appendChild(container);
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.background = 'white';
          }
        }, possibleSelectors);

        // Attendre un moment pour que les changements soient appliqués
        await waitRandomTime(500);

        // Générer le PDF
        await articlePage.pdf({
          path: filePath,
          format: 'A4',
          printBackground: true,
          margin: {
            top: '20mm',
            right: '20mm',
            bottom: '20mm',
            left: '20mm'
          }
        });

        console.log(`PDF sauvegardé: ${filename}`);

        // Ajouter l'article à la liste des téléchargés
        downloadedArticles.push({
          id: article.id,
          title: article.title,
          filename: filename,
          downloadedAt: new Date().toISOString()
        });

        // Sauvegarder la liste mise à jour
        fs.writeJSONSync(downloadedListFile, downloadedArticles, { spaces: 2 });

        // Log de progression
        fs.appendFileSync(logFile, `[${new Date().toISOString()}] Article téléchargé: ${article.title}\n`);

        // Attendre un peu avant de passer à l'article suivant
        await waitRandomTime(config.delays.download);
      } catch (error) {
        console.error(`Erreur lors du traitement de l'article ${article.title}: ${error.message}`);
        fs.appendFileSync(logFile, `[${new Date().toISOString()}] ERREUR: ${article.title} - ${error.message}\n`);
      }
    }

    console.log('Tous les articles ont été traités!');
    fs.appendFileSync(logFile, `[${new Date().toISOString()}] Fin du traitement - ${downloadedArticles.length}/${articles.length} articles téléchargés\n`);

  } catch (error) {
    console.error(`Erreur fatale: ${error.message}`);
    fs.appendFileSync(path.join(config.outputDir, 'log.txt'), `[${new Date().toISOString()}] ERREUR FATALE: ${error.message}\n`);
  } finally {
    // Fermer le navigateur
    await browser.close();
  }

  console.log('Script terminé!');
}

// Exécuter le script principal
main().catch(error => {
  console.error('Erreur non gérée:', error);
  process.exit(1);
});