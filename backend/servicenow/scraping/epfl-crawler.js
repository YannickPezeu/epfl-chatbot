// epfl-crawler-optimized.js
const puppeteer = require('puppeteer');
const fs = require('fs-extra');
const path = require('path');
const Queue = require('better-queue');
const dotenv = require('dotenv');
const http = require('http');
const https = require('https');

// Augmenter les limites de connexions HTTP
http.globalAgent.maxSockets = 15;
https.globalAgent.maxSockets = 15;

// Charger le fichier .env depuis le répertoire parent
const envPath = path.resolve(__dirname, '..', '.env');
const result = dotenv.config({ path: envPath });

if (result.error) {
  console.error(`Erreur lors du chargement du fichier .env: ${result.error.message}`);
  console.error(`Chemin tenté: ${envPath}`);
  process.exit(1);
}

// Vérifier que les identifiants sont disponibles
if (!process.env.EPFL_USERNAME || !process.env.EPFL_PASSWORD) {
  console.error('Erreur: Les identifiants EPFL_USERNAME et EPFL_PASSWORD doivent être définis dans le fichier .env');
  process.exit(1);
}

console.log(`Fichier .env chargé avec succès depuis: ${envPath}`);

// Configuration
const config = {
  baseUrl: 'https://www.epfl.ch',
  outputDir: path.join(__dirname, 'epfl_data'),
  maxDepth: 3,
  credentials: {
    username: process.env.EPFL_USERNAME,
    password: process.env.EPFL_PASSWORD
  },
  concurrency: 1,  // Réduire à 1 pour éviter de surcharger le serveur
  documentConcurrency: 1,
  delayBetweenRequests: 10000, // 10 secondes entre chaque requête
  downloadTimeout: 60000, // 60 secondes pour les téléchargements
  maxRetries: 3, // Nombre de tentatives pour les téléchargements de documents
  // Ajouter un délai aléatoire pour paraître plus "humain"
  randomDelay: true, // Activer un délai aléatoire
  minDelay: 8000,    // Délai minimum (8 secondes)
  maxDelay: 15000,   // Délai maximum (15 secondes)
  documentExtensions: ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.txt'],
  allowedDomains: ['epfl.ch', 'www.epfl.ch', 'actu.epfl.ch', 'inside.epfl.ch'],
  // Domaines à ignorer (souvent sources d'erreurs ou hors scope)
  ignoreDomains: ['social.epfl.ch', 'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com']
};

// Rotation des user agents
const userAgents = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
];

// URLs à explorer
const startUrls = [
  // 'https://www.epfl.ch',
  // 'https://www.epfl.ch/research/',
  // 'https://www.epfl.ch/education/',
  // 'https://www.epfl.ch/campus/',
  // 'https://actu.epfl.ch',
  // 'https://inside.epfl.ch',  // Nécessite authentification
  'https://support.epfl.ch/epfl?id=epfl_kb_home',
];

// Initialiser l'état
let visited = new Set();
let planned = new Set(); // Pour éviter d'ajouter plusieurs fois la même URL
let documentQueue = [];
let failedDownloads = []; // Pour garder trace des documents qui n'ont pas pu être téléchargés
let totalPages = 0;
let totalDocuments = 0;
let linksFound = 0;
let pagesProcessed = 0;
const resetThreshold = 50; // Réinitialiser après 50 pages

// Créer les dossiers de sortie
fs.ensureDirSync(config.outputDir);
fs.ensureDirSync(path.join(config.outputDir, 'pages'));
fs.ensureDirSync(path.join(config.outputDir, 'documents'));

// Version compatible de waitForTimeout
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// Fonction pour obtenir un délai aléatoire
function getRandomDelay() {
  if (!config.randomDelay) {
    return config.delayBetweenRequests;
  }
  return Math.floor(Math.random() * (config.maxDelay - config.minDelay + 1)) + config.minDelay;
}

// Fonction pour obtenir un User-Agent aléatoire
function getRandomUserAgent() {
  return userAgents[Math.floor(Math.random() * userAgents.length)];
}

// Fonction pour obtenir un chemin de fichier correspondant à la structure du site
function getPathFromUrl(url, baseDir, defaultExtension = '.html') {
  try {
    const { hostname, pathname } = new URL(url);

    // Créer un chemin basé sur le hostname et le pathname
    let urlPath = pathname;

    // Traiter les chemins qui se terminent par un slash
    if (urlPath.endsWith('/')) {
      urlPath += 'index' + defaultExtension;
    } else if (!path.extname(urlPath)) {
      // Ajouter index.html pour les chemins sans extension
      urlPath += '/index' + defaultExtension;
    }

    // Si le hostname est différent du domaine principal, l'inclure dans le chemin
    let dirPath;
    if (hostname === 'www.epfl.ch') {
      dirPath = urlPath;
    } else {
      // Remplacer les points par des underscores pour éviter des problèmes de chemin
      const hostDir = hostname.replace(/\./g, '_');
      dirPath = path.join(hostDir, urlPath);
    }

    // Nettoyer le chemin pour éviter les caractères problématiques
    dirPath = dirPath.replace(/[?#]/g, '_');

    return path.join(baseDir, dirPath);
  } catch (error) {
    // Fallback en cas d'URL invalide
    console.error(`Erreur lors de la création du chemin pour ${url}: ${error.message}`);
    return path.join(baseDir, 'unknown', `page_${Date.now()}${defaultExtension}`);
  }
}

// Fonction pour réinitialiser le navigateur périodiquement
async function resetBrowser(browser) {
  try {
    console.log("Réinitialisation du navigateur...");
    await browser.close();

    // Pause avant de redémarrer
    await delay(20000);

    // Relancer le navigateur avec les mêmes options
    return await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
        '--window-size=1920x1080',
      ],
      defaultViewport: { width: 1920, height: 1080 }
    });
  } catch (error) {
    console.error(`Erreur lors de la réinitialisation du navigateur: ${error.message}`);
    // Si la réinitialisation échoue, créer un nouveau navigateur
    return await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
      ],
      defaultViewport: { width: 1366, height: 768 }
    });
  }
}

// Fonction plus robuste pour récupérer le contenu de la page
async function safeGetContent(page) {
  try {
    return await page.content();
  } catch (error) {
    console.error(`Erreur lors de la récupération du contenu: ${error.message}`);
    // Tenter une approche alternative
    try {
      return await page.evaluate(() => document.documentElement.outerHTML);
    } catch (innerError) {
      console.error(`Échec de la récupération alternative: ${innerError.message}`);
      return "<html><body><p>Échec de la récupération du contenu</p></body></html>";
    }
  }
}

// Fonction pour sauvegarder un point de contrôle
function saveCheckpoint() {
  const checkpoint = {
    timestamp: new Date().toISOString(),
    visited: Array.from(visited),
    planned: Array.from(planned),
    documentQueue: documentQueue,
    totalPages,
    totalDocuments,
    linksFound,
    pagesProcessed
  };

  try {
    fs.writeFileSync(
      path.join(config.outputDir, 'checkpoint.json'),
      JSON.stringify(checkpoint, null, 2)
    );
    console.log("Point de contrôle sauvegardé");
  } catch (error) {
    console.error(`Erreur lors de la sauvegarde du point de contrôle: ${error.message}`);
  }
}

// Fonction pour charger un point de contrôle
function loadCheckpoint() {
  const checkpointPath = path.join(config.outputDir, 'checkpoint.json');

  if (fs.existsSync(checkpointPath)) {
    try {
      const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));

      // Restaurer l'état
      visited = new Set(checkpoint.visited);
      planned = new Set(checkpoint.planned);
      documentQueue = checkpoint.documentQueue;
      totalPages = checkpoint.totalPages || 0;
      totalDocuments = checkpoint.totalDocuments || 0;
      linksFound = checkpoint.linksFound || 0;
      pagesProcessed = checkpoint.pagesProcessed || 0;

      console.log(`Point de contrôle chargé: ${checkpoint.timestamp}`);
      console.log(`URLs visitées restaurées: ${visited.size}`);
      console.log(`Documents en attente: ${documentQueue.length}`);

      return true;
    } catch (error) {
      console.error(`Erreur lors du chargement du point de contrôle: ${error.message}`);
    }
  }

  return false;
}

// Fonction principale
async function main() {
  // Vérifier si reprise depuis un checkpoint
  const resumeFromCheckpoint = process.argv.includes('--resume');
  if (resumeFromCheckpoint) {
    const loaded = loadCheckpoint();
    if (!loaded) {
      console.log("Aucun point de contrôle trouvé, démarrage d'un nouveau crawl");
    }
  }

  console.log('Démarrage du crawler EPFL avec Puppeteer...');
  console.log(`Utilisation du compte: ${config.credentials.username}`);

  // Lancer le navigateur avec des options robustes
  let browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--disable-gpu',
      '--window-size=1920x1080',
    ],
    defaultViewport: { width: 1920, height: 1080 }
  });

  try {
    // Créer une page pour l'authentification
    const authPage = await browser.newPage();

    // Configurer la navigation
    await setupPage(authPage);

    // S'authentifier
    await authenticate(authPage);

    // Fermer la page d'authentification
    await authPage.close();

    // Sauvegarder périodiquement la progression
    const saveInterval = setInterval(() => {
      saveProgress();
    }, 5 * 60 * 1000);

    // Sauvegarder périodiquement les points de contrôle
    const checkpointInterval = setInterval(() => {
      saveCheckpoint();
    }, 15 * 60 * 1000);

    // Créer la file d'attente de pages à visiter
    const pageQueue = new Queue(async (url, cb) => {
      try {
        const { href, depth, retries = 0 } = url;

        // Si l'URL a déjà été visitée ou si la profondeur est trop élevée, ignorer
        if (visited.has(href) || depth > config.maxDepth) {
          cb(null);
          return;
        }

        // Vérifier si l'URL doit être ignorée
        if (shouldIgnoreUrl(href)) {
          cb(null);
          return;
        }

        // Créer une nouvelle page
        const page = await browser.newPage();
        await setupPage(page);

        try {
          // Crawler la page
          await crawlPage(page, href, depth, pageQueue, browser);

          // Incrémenter le compteur
          pagesProcessed++;

          // Vérifier si une réinitialisation est nécessaire
          if (pagesProcessed >= resetThreshold) {
            browser = await resetBrowser(browser);
            pagesProcessed = 0;
          }

          // Pause après chaque 10 pages
          if (pagesProcessed % 10 === 0) {
            console.log("Pause de repos après 10 pages...");
            await delay(20000); // Pause de 20 secondes
          }

        } catch (error) {
          console.error(`Erreur lors du crawl de ${href}: ${error.message}`);

          // Si erreur de connexion, faire une pause plus longue
          if (error.message.includes("Connection closed") ||
              error.message.includes("frame was detached") ||
              error.message.includes("Protocol error")) {
            console.log("Pause plus longue après erreur de connexion...");
            await delay(30000); // Pause de 30 secondes
          }

          // Réessayer si nécessaire
          if (retries < 2) {
            console.log(`Nouvelle tentative pour ${href} (${retries + 1}/2)`);
            pageQueue.push({ href, depth, retries: retries + 1 });
          }
        } finally {
          // Fermer la page dans tous les cas
          await page.close();
        }

        cb(null);
      } catch (error) {
        console.error(`Erreur lors du traitement de ${url.href}: ${error.message}`);
        cb(error);
      }
    }, {
      concurrent: config.concurrency,
      afterProcessDelay: getRandomDelay,
      maxRetries: 2,
      retryDelay: 10000
    });

    // Créer la file d'attente de documents à télécharger avec une méthode robuste
    const downloadQueue = new Queue(async (docInfo, cb) => {
      try {
        const success = await downloadDocumentWithNode(docInfo.url, docInfo.filePath, docInfo.retries || 0);
        if (!success && (docInfo.retries || 0) < config.maxRetries) {
          // Réessayer plus tard avec un délai progressif
          const nextRetry = (docInfo.retries || 0) + 1;
          const delay = nextRetry * 5000; // 5s, 10s, 15s...

          console.log(`Échec du téléchargement, nouvelle tentative ${nextRetry}/${config.maxRetries} pour ${docInfo.url} dans ${delay/1000}s`);

          setTimeout(() => {
            downloadQueue.push({
              url: docInfo.url,
              filePath: docInfo.filePath,
              retries: nextRetry
            });
          }, delay);
        } else if (!success) {
          failedDownloads.push(docInfo.url);
        }

        cb(null);
      } catch (error) {
        console.error(`Erreur lors du téléchargement de ${docInfo.url}: ${error.message}`);

        // Ajouter aux téléchargements échoués après nombre max de tentatives
        if ((docInfo.retries || 0) >= config.maxRetries) {
          failedDownloads.push(docInfo.url);
        }

        cb(error);
      }
    }, {
      concurrent: config.documentConcurrency,
      afterProcessDelay: getRandomDelay,
      maxRetries: 1,
      retryDelay: 10000
    });

    // Ajouter les URLs de départ à la file d'attente
    if (!resumeFromCheckpoint) {
      for (const url of startUrls) {
        if (!planned.has(url)) {
          planned.add(url);
          pageQueue.push({ href: url, depth: 0 });
        }
      }
    }

    // Attendre que toutes les pages soient traitées
    await new Promise((resolve) => {
      pageQueue.on('drain', () => {
        console.log('Toutes les pages ont été traitées.');
        console.log(`Total des liens trouvés: ${linksFound}`);
        console.log(`Pages visitées: ${visited.size}`);
        console.log(`Documents à télécharger: ${documentQueue.length}`);

        // Arrêter les intervalles de sauvegarde
        clearInterval(saveInterval);
        clearInterval(checkpointInterval);

        // Sauvegarder une dernière fois
        saveProgress();
        saveCheckpoint();

        // Ajouter tous les documents à la file de téléchargement
        for (const doc of documentQueue) {
          downloadQueue.push(doc);
        }

        // Attendre que tous les documents soient téléchargés
        downloadQueue.on('drain', () => {
          console.log('Tous les documents ont été téléchargés.');
          console.log(`Documents échoués: ${failedDownloads.length}`);

          // Sauvegarder la liste des documents qui ont échoué
          fs.writeFileSync(
            path.join(config.outputDir, 'failed_downloads.txt'),
            failedDownloads.join('\n')
          );

          resolve();
        });

        // Si aucun document n'est trouvé, résoudre immédiatement
        if (documentQueue.length === 0) {
          resolve();
        }
      });
    });

    // Générer un rapport
    generateReport();

  } finally {
    // Fermer le navigateur
    await browser.close();
  }

  console.log('Crawling terminé!');
}

// Configurer les options de la page
async function setupPage(page) {
  // Configurer le viewport
  await page.setViewport({ width: 1366, height: 768 });

  // Configurer les en-têtes avec un user agent aléatoire
  await page.setExtraHTTPHeaders({
    'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
    'User-Agent': getRandomUserAgent()
  });

  // Augmenter les timeouts
  await page.setDefaultNavigationTimeout(45000);
  await page.setDefaultTimeout(30000);

  // Intercepter les requêtes pour les médias et les stylesheets (optionnel pour accélérer)
  await page.setRequestInterception(true);
  page.on('request', (request) => {
    const resourceType = request.resourceType();
    // Ignorer les ressources non nécessaires pour le crawling
    if (['image', 'stylesheet', 'font', 'media'].includes(resourceType)) {
      request.abort();
    } else {
      request.continue();
    }
  });

  // Ignorer la plupart des erreurs de console
  page.on('console', msg => {
    if (msg.type() === 'error' &&
        !msg.text().includes('favicon.ico') &&
        !msg.text().includes('Failed to load resource') &&
        !msg.text().includes('Content-Security-Policy') &&
        !msg.text().includes('Refused to frame') &&
        !msg.text().includes('Refused to execute script') &&
        !msg.text().includes('[Report Only]')) {
      console.log(`Erreur console importante: ${msg.text()}`);
    }
  });
}

// Fonction d'authentification
async function authenticate(page) {
  try {
    console.log('Tentative d\'authentification...');

    // Accéder à une page qui nécessite une authentification
    await page.goto('https://inside.epfl.ch', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    // Vérifier si nous sommes sur la page de login
    if (page.url().includes('tequila.epfl.ch')) {
      console.log('Page de login détectée, authentification en cours...');

      // Attendre que le formulaire de login soit chargé
      await page.waitForSelector('#username', { timeout: 10000 });

      // Remplir le formulaire
      await page.type('#username', config.credentials.username);
      await page.type('#password', config.credentials.password);

      // Soumettre et attendre la redirection
      await Promise.all([
        page.click('button[type="submit"]'),
        page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 })
      ]);

      // Vérifier si l'authentification a réussi
      if (!page.url().includes('tequila.epfl.ch')) {
        console.log('Authentification réussie!');

        // Sauvegarder les cookies pour les utiliser dans d'autres pages
        const cookies = await page.cookies();
        await fs.writeJSON(path.join(config.outputDir, 'cookies.json'), cookies);
      } else {
        throw new Error('Échec de l\'authentification. Vérifiez vos identifiants.');
      }
    } else {
      console.log('Déjà connecté ou authentification non requise.');
    }
  } catch (error) {
    console.error(`Erreur lors de l'authentification: ${error.message}`);
    throw error;
  }
}

// Fonction pour vérifier si une URL doit être ignorée
function shouldIgnoreUrl(url) {
  try {
    const { hostname, pathname } = new URL(url);

    // Vérifier si le domaine est dans la liste des domaines à ignorer
    if (config.ignoreDomains.some(domain => hostname.includes(domain))) {
      return true;
    }

    // Ignorer les URLs de protocoles non standards
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      return true;
    }

    // Vérifier si l'URL est dans un domaine autorisé
    if (!config.allowedDomains.some(domain => hostname.endsWith(domain))) {
      return true;
    }

    return false;
  } catch (error) {
    // Si l'URL est invalide, l'ignorer
    return true;
  }
}

// Fonction de crawling (avec queue et browser en paramètres)
async function crawlPage(page, url, depth, queue, browser) {
  // Si l'URL a déjà été visitée, ignorer
  if (visited.has(url)) {
    return;
  }

  // Marquer l'URL comme visitée
  visited.add(url);

  console.log(`Crawling [${depth}]: ${url}`);

  try {
    // Naviguer vers la page avec une gestion d'erreur robuste
    const response = await page.goto(url, {
      waitUntil: 'domcontentloaded', // Plus rapide que 'networkidle2'
      timeout: 30000
    });

    // Attendre un peu plus pour que les éléments se chargent (délai aléatoire)
    await delay(getRandomDelay());

    // Vérifier si la page existe et est accessible
    if (!response) {
      console.log(`Pas de réponse pour ${url}`);
      return;
    }

    const status = response.status();
    if (status !== 200) {
      console.log(`Statut HTTP ${status} pour ${url}`);

      // Gérer les redirections (300-399)
      if (status >= 300 && status < 400) {
        const redirectUrl = response.headers().location;
        if (redirectUrl && !visited.has(redirectUrl) && !planned.has(redirectUrl)) {
          planned.add(redirectUrl);
          queue.push({ href: redirectUrl, depth });
        }
      }

      return;
    }

    // Vérifier le type de contenu
    const contentType = response.headers()['content-type'] || '';
    if (!contentType.includes('text/html')) {
      // Si c'est un document, l'ajouter à la file de téléchargement
      if (isDocumentUrl(url)) {
        const documentsDir = path.join(config.outputDir, 'documents');
        const filePath = getPathFromUrl(url, documentsDir);

        documentQueue.push({
          url,
          filePath,
          sourceUrl: url
        });
      }
      return;
    }

    // Obtenir le chemin hiérarchique pour cette page
    const pagesDir = path.join(config.outputDir, 'pages');
    const filePath = getPathFromUrl(url, pagesDir);
    const dirPath = path.dirname(filePath);

    // Créer les répertoires nécessaires
    fs.ensureDirSync(dirPath);

    // Sauvegarder la page HTML (avec méthode robuste)
    const pageId = (++totalPages).toString().padStart(6, '0');
    const pageContent = await safeGetContent(page);

    // Utiliser le nom de fichier du chemin calculé
    const filename = path.basename(filePath);

    const metadata = {
      url,
      title: await page.title(),
      fetchedAt: new Date().toISOString(),
      depth,
      savedPath: path.relative(pagesDir, filePath)
    };

    await Promise.all([
      fs.writeFile(filePath, pageContent),
      fs.appendFile(
        path.join(config.outputDir, 'index.txt'),
        `${url}\t${metadata.savedPath}\t${metadata.title}\n`
      )
    ]);

    // Extraire tous les liens de manière robuste
    const links = await page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll('a[href]'));
      return anchors.map(a => a.href).filter(href => href && href.trim() !== '');
    });

    // Log pour debug
    console.log(`Trouvé ${links.length} liens sur cette page`);
    linksFound += links.length;

    // Traiter chaque lien
    for (const link of links) {
      try {
        if (!link) continue;

        // Nettoyer l'URL
        const cleanLink = cleanUrl(link);
        if (!cleanLink) continue;

        // Ignorer les URLs qui doivent être ignorées
        if (shouldIgnoreUrl(cleanLink)) continue;

        // Vérifier si c'est un document
        if (isDocumentUrl(cleanLink)) {
          const documentsDir = path.join(config.outputDir, 'documents');
          const docPath = getPathFromUrl(cleanLink, documentsDir);

          if (!documentQueue.some(d => d.url === cleanLink)) {
            documentQueue.push({
              url: cleanLink,
              filePath: docPath,
              sourceUrl: url
            });
          }
        }
        // Sinon, ajouter à la file des pages à visiter (si pas encore planifié)
        else if (!visited.has(cleanLink) && !planned.has(cleanLink)) {
          planned.add(cleanLink); // Marquer comme planifié pour éviter les doublons
          queue.push({ href: cleanLink, depth: depth + 1 });
        }
      } catch (error) {
        console.error(`Erreur lors du traitement du lien ${link}: ${error.message}`);
      }
    }
  } catch (error) {
    console.error(`Erreur lors du crawling de ${url}: ${error.message}`);
    throw error; // Relancer pour que la queue puisse gérer les tentatives
  }
}

// Télécharger un document avec Node.js natif (sans puppeteer)
async function downloadDocumentWithNode(url, filePath, retryCount = 0) {
  // Vérifier si l'URL est valide
  if (!url || !isDocumentUrl(url)) {
    return false;
  }

  console.log(`Téléchargement du document (tentative ${retryCount + 1}): ${url}`);

  try {
    // Créer les répertoires nécessaires
    const dirPath = path.dirname(filePath);
    fs.ensureDirSync(dirPath);

    // Si le fichier existe déjà, le considérer comme téléchargé
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      if (stats.size > 0) {
        console.log(`Document déjà téléchargé: ${path.basename(filePath)}`);
        totalDocuments++;
        return true;
      }
    }

    // Créer un fichier temporaire
    const tempFilePath = `${filePath}.tmp`;

    // Télécharger le fichier avec un timeout
    return new Promise((resolve) => {
      const fileStream = fs.createWriteStream(tempFilePath);

      const protocol = url.startsWith('https') ? https : http;
      const request = protocol.get(url, {
        timeout: config.downloadTimeout,
        headers: {
          'User-Agent': getRandomUserAgent(),
          'Accept': 'application/pdf,application/msword,application/vnd.openxmlformats-officedocument.*,*/*'
        }
      }, (response) => {
        // Gérer les redirections manuellement
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          // Fermer le stream actuel
          fileStream.end();

          // Construire l'URL absolue si nécessaire
          let redirectUrl = response.headers.location;
          if (!redirectUrl.startsWith('http')) {
            const baseUrl = new URL(url);
            redirectUrl = new URL(redirectUrl, `${baseUrl.protocol}//${baseUrl.host}`).toString();
          }

          // Appeler récursivement avec la nouvelle URL
          console.log(`Redirection vers ${redirectUrl}`);
          downloadDocumentWithNode(redirectUrl, filePath, retryCount)
            .then(resolve)
            .catch(() => resolve(false));
          return;
        }

        // Vérifier que la réponse est valide
        if (response.statusCode !== 200) {
          console.error(`Statut HTTP ${response.statusCode} pour ${url}`);
          fileStream.end();
          resolve(false);
          return;
        }

        // Pipe la réponse vers le fichier
        response.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close();

          // Vérifier que le fichier a été téléchargé correctement
          try {
            const stats = fs.statSync(tempFilePath);
            if (stats.size === 0) {
              console.error(`Fichier vide téléchargé pour ${url}`);
              fs.unlinkSync(tempFilePath);
              resolve(false);
              return;
            }

            // Renommer le fichier temporaire en fichier final
            fs.renameSync(tempFilePath, filePath);

            totalDocuments++;
            console.log(`Document téléchargé avec succès: ${path.basename(filePath)} (${formatFileSize(stats.size)})`);

            // Enregistrer le document dans l'index
            fs.appendFileSync(
              path.join(config.outputDir, 'documents_index.txt'),
              `${url}\t${path.relative(path.join(config.outputDir, 'documents'), filePath)}\n`
            );

            // Pause après chaque téléchargement
            setTimeout(() => {
              resolve(true);
            }, getRandomDelay());
          } catch (err) {
            console.error(`Erreur lors de la vérification du fichier ${url}: ${err.message}`);
            try { fs.unlinkSync(tempFilePath); } catch {}
            resolve(false);
          }
        });
      });

      request.on('error', (err) => {
        console.error(`Erreur réseau lors du téléchargement de ${url}: ${err.message}`);
        fileStream.close();
        try { fs.unlinkSync(tempFilePath); } catch {}
        resolve(false);
      });

      // Définir un timeout global
      request.setTimeout(config.downloadTimeout, () => {
        console.error(`Timeout lors du téléchargement de ${url}`);
        request.abort();
        fileStream.close();
        try { fs.unlinkSync(tempFilePath); } catch {}
        resolve(false);
      });
    });

  } catch (error) {
    console.error(`Erreur lors du téléchargement de ${url}: ${error.message}`);
    return false;
  }
}

// Sauvegarder l'état actuel du crawling
function saveProgress() {
  try {
    console.log("Sauvegarde de la progression...");

    // Sauvegarder les URLs visitées
    fs.writeFileSync(
      path.join(config.outputDir, 'visited_urls.txt'),
      Array.from(visited).join('\n')
    );

    // Sauvegarder les documents en attente
    fs.writeFileSync(
      path.join(config.outputDir, 'pending_documents.txt'),
      documentQueue.map(doc => doc.url).join('\n')
    );

    // Sauvegarder les statistiques
    const stats = {
      timestamp: new Date().toISOString(),
      pagesVisited: visited.size,
      pagesSaved: totalPages,
      documentsDownloaded: totalDocuments,
      documentsPending: documentQueue.length,
      linksFound: linksFound
    };

    fs.writeJSONSync(
      path.join(config.outputDir, 'progress.json'),
      stats,
      { spaces: 2 }
    );

    console.log("Progression sauvegardée");
  } catch (error) {
    console.error(`Erreur lors de la sauvegarde de la progression: ${error.message}`);
  }
}

// Formater la taille du fichier de manière lisible
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

// Vérifier si une URL correspond à un document
function isDocumentUrl(url) {
  if (!url) return false;

  try {
    const { pathname } = new URL(url);
    const extension = path.extname(pathname).toLowerCase();
    return config.documentExtensions.includes(extension);
  } catch (error) {
    return false;
  }
}

// Nettoyer une URL (supprimer les fragments, etc.)
function cleanUrl(url) {
  try {
    // Ignorer les URLs javascript: et mailto:
    if (url.startsWith('javascript:') || url.startsWith('mailto:') || url.startsWith('tel:')) {
      return null;
    }

    const parsed = new URL(url);
    parsed.hash = '';

    // Supprimer les paramètres de tracking courants
    const paramsToRemove = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid'];
    paramsToRemove.forEach(param => parsed.searchParams.delete(param));

    return parsed.toString();
  } catch (error) {
    return null;
  }
}

// Générer un rapport final
function generateReport() {
  const report = {
    timestamp: new Date().toISOString(),
    statistics: {
      pagesVisited: visited.size,
      pagesSaved: totalPages,
      documentsDownloaded: totalDocuments,
      documentsFailed: failedDownloads.length,
      linksFound: linksFound
    },
    config: {
      ...config,
      credentials: {
        username: config.credentials.username,
        password: '***' // Ne pas inclure le mot de passe dans le rapport
      }
    }
  };

  fs.writeJSONSync(
    path.join(config.outputDir, 'rapport.json'),
    report,
    { spaces: 2 }
  );

  console.log('Rapport généré:');
  console.log(`Pages visitées: ${report.statistics.pagesVisited}`);
  console.log(`Pages sauvegardées: ${report.statistics.pagesSaved}`);
  console.log(`Documents téléchargés: ${report.statistics.documentsDownloaded}`);
  console.log(`Documents échoués: ${report.statistics.documentsFailed}`);
  console.log(`Liens trouvés: ${report.statistics.linksFound}`);
}

// Exécuter le script principal
main().catch(error => {
  console.error('Erreur fatale:', error);
  saveProgress(); // Sauvegarde d'urgence en cas d'erreur
  saveCheckpoint();
  process.exit(1);
});