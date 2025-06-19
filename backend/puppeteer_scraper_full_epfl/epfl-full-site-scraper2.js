// epfl-full-site-scraper.js
const puppeteer = require('puppeteer');
const fs = require('fs-extra');
const path = require('path');
const dotenv = require('dotenv');
const { setTimeout } = require('timers/promises');
const crypto = require('crypto');
const { URL } = require('url');
const robotsParser = require('robots-parser');
const https = require('https'); // Add this at the top of your file if not already there
const http = require('http');  // In case of non-https redirects (less likely for final PDF)

// Load environment variables
const envPath = path.resolve(__dirname, '../.env');
dotenv.config({ path: envPath });

// Check credentials
if (!process.env.EPFL_USERNAME_TEQUILA || !process.env.EPFL_USERNAME_MICROSOFT || !process.env.EPFL_PASSWORD) {
  console.error('Error: EPFL_USERNAME_TEQUILA, EPFL_USERNAME_MICROSOFT, and EPFL_PASSWORD must be defined in .env file');
  process.exit(1);
}

// Configuration
const config = {
  baseUrls: [
      // 'https://www.epfl.ch/campus/services/human-resources/wp-content/uploads/2022/08/OPers_English.pdf',
      // 'https://www.epfl.ch/campus/services/human-resources/laws-and-regulations-lex/',
      // 'https://www.epfl.ch/research/solutions-for-sustainability-initiative-s4s',
    'https://inside.epfl.ch',
    'https://www.epfl.ch',
    'https://support.epfl.ch',
    'https://wiki.epfl.ch',
    'https://memento.epfl.ch',
  ],
  outputDir: path.join(__dirname, 'epfl_full_site_data2'),
  credentials: { // (UPDATED)
    usernameTequila: process.env.EPFL_USERNAME_TEQUILA,
    usernameMicrosoft: process.env.EPFL_USERNAME_MICROSOFT,
    password: process.env.EPFL_PASSWORD
  },
  crawling: {
    maxDepth: 10,
    maxPages: 100000,
    concurrentPages: 5,
    respectRobotsTxt: true,
    relaunchBrowserAfterPages: 100, // Relaunch browser after this many pages
    relaunchBrowserAfterHours: 2,   // Relaunch browser after this many hours
    allowedDomains: [
      'epfl.ch',
      'support.epfl.ch',
      'wiki.epfl.ch',
      'memento.epfl.ch',
      'inside.epfl.ch'
    ],
    allowedPathPrefixes: [ // New setting for specific paths
        // 'https://www.epfl.ch/campus/services/human-resources/laws-and-regulations-lex/'
      // 'https://www.epfl.ch/research/solutions-for-sustainability-initiative-s4s/'
      // You can add more prefixes if needed
    ],
    excludePatterns: [
      /^https?:\/\/plan\.epfl\.ch/i,
      /^https?:\/\/infoscience\.epfl\.ch/i,
      /^https?:\/\/actu\.epfl\.ch/i,
      /\.(jpg|jpeg|png|gif|svg|ico|webp|bmp)$/i,
      /\.(mp4|avi|mov|wmv|flv|webm)$/i,
      /\.(mp3|wav|ogg|flac)$/i,
      /\.(zip|rar|7z|tar|gz)$/i,
      /\/logout/i,
      /\/signout/i,
      /\/deconnexion/i,
      /mailto:/i,
      /tel:/i,
    ],
    includeFileTypes: [ // These are treated as downloadable documents
      '.pdf',
      '.doc',
      '.docx',
      '.ppt',
      '.pptx',
      '.xls',
      '.xlsx'
    ]
  },
  delays: {
    navigation: 3000,
    scroll: 500,
    typing: 100, // Slightly increased for reliability
    download: 5000, // Increased for larger files
    randomMin: 500, // Increased base random delay
    randomMax: 2000,
    rateLimitDelay: 1000
  },
  userAgents: [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36', // Updated
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15', // Updated
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0' // Updated
  ],
  contentExtraction: {
    selectors: {
      main: ['main', '#main', '.main-content', '#content', '.content'],
      article: ['article', '.article', '.post', '.entry-content'],
      // Remove these elements text content for RAG, but their links will still be crawled
      remove: [
        'script', 'style', 'noscript', 'iframe', 'header', 'footer', 'nav',
        '.advertisement', '#cookie-banner', '.cookie-notice', '.privacy-banner',
        '.header', '.footer', '#header', '#footer', '.navbar', '.nav-menu',
        'aside', '.sidebar', '#sidebar', '#menu-main', // Added problematic menu
        '.breadcrumb', '.breadcrumbs', '.site-map', // Common non-content navigational elements
      ]
    }
  }
};

// Global for robots.txt parsers
let robotsCache = {};
let lastBrowserLaunchTime = Date.now();
let pagesSinceLastBrowserLaunch = 0;

// Utilities
function getRandomDelay() {
  return Math.floor(Math.random() * (config.delays.randomMax - config.delays.randomMin + 1)) + config.delays.randomMin;
}

function getRandomUserAgent() {
  return config.userAgents[Math.floor(Math.random() * config.userAgents.length)];
}

async function waitRandomTime(baseTime = 0) {
  const randomTime = getRandomDelay();
  await setTimeout(baseTime + randomTime);
}

function normalizeUrl(urlStr) {
  try {
    const urlObj = new URL(urlStr);
    urlObj.hash = ''; // Remove fragments
    urlObj.searchParams.sort(); // Normalize query parameters
    return urlObj.toString();
  } catch (e) {
    // console.warn(`Invalid URL for normalization: ${urlStr}`, e);
    return urlStr; // Return original if invalid
  }
}

async function getRobots(url) {
  const urlObj = new URL(url);
  const robotsUrl = `${urlObj.protocol}//${urlObj.hostname}/robots.txt`;

  if (robotsCache[robotsUrl]) {
    return robotsCache[robotsUrl];
  }

  try {
    // Use a simple fetch, or a new Puppeteer page if complex JS is needed for robots.txt (rare)
    const response = await fetch(robotsUrl, { headers: { 'User-Agent': getRandomUserAgent() }});
    if (response.ok) {
      const robotsTxt = await response.text();
      robotsCache[robotsUrl] = robotsParser(robotsUrl, robotsTxt);
      return robotsCache[robotsUrl];
    } else {
      console.warn(`Failed to fetch ${robotsUrl}: ${response.status}`);
      robotsCache[robotsUrl] = robotsParser(robotsUrl, ''); // Assume allow all if robots.txt is missing/error
      return robotsCache[robotsUrl];
    }
  } catch (error) {
    console.error(`Error fetching robots.txt for ${urlObj.hostname}:`, error);
    robotsCache[robotsUrl] = robotsParser(robotsUrl, ''); // Assume allow all on error
    return robotsCache[robotsUrl];
  }
}

async function isAllowedByRobots(url) {
  if (!config.crawling.respectRobotsTxt) {
    return true;
  }
  try {
    const robots = await getRobots(url);
    return robots.isAllowed(url, getRandomUserAgent()); // Check against one of our UAs
  } catch (error) {
    console.error(`Error checking robots.txt for ${url}:`, error);
    return true; // Fail open: allow if robots check fails
  }
}

function isAllowedUrl(url, currentUrl = null) {
  try {
    // Ensure URL is absolute using the current page's URL as base if necessary
    const absoluteUrlStr = currentUrl ? new URL(url, currentUrl).toString() : new URL(url).toString();
    const urlObj = new URL(absoluteUrlStr);

    // 1. Check against allowedDomains (hostname check)
    // Ensures we are on an approved TLD or subdomain.
    const isDomainAllowed = config.crawling.allowedDomains.some(domain =>
      urlObj.hostname === domain || urlObj.hostname.endsWith('.' + domain)
    );
    if (!isDomainAllowed) {
      // console.log(`URL rejected (domain not allowed): ${absoluteUrlStr}`);
      return false;
    }

    // 2. Check against allowedPathPrefixes (if defined and not empty)
    // Restricts crawling to specific starting paths.
    if (config.crawling.allowedPathPrefixes && config.crawling.allowedPathPrefixes.length > 0) {
      const matchesPathPrefix = config.crawling.allowedPathPrefixes.some(prefix =>
        absoluteUrlStr.startsWith(prefix)
      );
      if (!matchesPathPrefix) {
        // console.log(`URL rejected (path prefix not allowed): ${absoluteUrlStr}`);
        return false;
      }
    }

    // 3. Check exclude patterns
    // Rejects URLs matching specific regex patterns.
    const isExcludedByPattern = config.crawling.excludePatterns.some(pattern =>
      pattern.test(absoluteUrlStr)
    );
    if (isExcludedByPattern) {
      // console.log(`URL rejected (excluded by pattern): ${absoluteUrlStr}`);
      return false;
    }

    // 4. Handle includeFileTypes (these should bypass some generic extension exclusions if explicitly included)
    // This logic mostly applies to whether a URL is considered downloadable vs. crawlable for HTML content.
    // The primary role here is to ensure that if it *is* an included file type, it doesn't get accidentally
    // rejected by overly broad subsequent checks if those were to be added.
    const urlPath = urlObj.pathname.toLowerCase();
    const isIncludedFileType = config.crawling.includeFileTypes.some(ext => urlPath.endsWith(ext));
    if (isIncludedFileType) {
      return true; // If it's an explicitly included file type, it's allowed (domain/path checks passed)
    }

    // 5. Optional: Further exclude if it looks like a non-content file not in includeFileTypes
    // This can be a bit aggressive. Example: reject any URL with an extension not .html/.htm or in includeFileTypes
    if (/\.[a-z0-9]{2,5}$/i.test(urlObj.pathname) && // has some common extension length
        !isIncludedFileType &&
        !urlPath.endsWith('.html') &&
        !urlPath.endsWith('.htm') &&
        urlObj.pathname.split('/').pop().includes('.') // last path segment has a dot (likely a file)
       ) {
      // console.log(`URL considered non-HTML/non-included document, rejected: ${absoluteUrlStr}`);
      // return false; // Uncomment this line if you want to be very strict about non-document file types
    }

    return true; // If all checks pass
  } catch (e) {
    // console.warn(`Invalid URL encountered in isAllowedUrl ("${url}", base "${currentUrl}"): ${e.message}`);
    return false;
  }
}

function generateFileNameFromUrl(url) {
  const urlObj = new URL(url);
  let pathPart = urlObj.pathname.substring(1).replace(/\/$/, ''); // Remove leading/trailing slashes
  if (!pathPart) pathPart = 'index';
  const fileName = pathPart.split('/').filter(Boolean).join('_') || 'index';
  const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
  return `${fileName}_${hash}`.replace(/[^a-z0-9_-]/gi, '_').substring(0, 100);
}

// Crawler State Management
class CrawlerState {
  constructor(stateFile) {
    this.stateFile = stateFile;
    this.visited = new Set();
    this.toVisit = new Map();
    this.failed = new Map();
    this.content = new Map(); // Stores { url, title, text, links, metadata, timestamp }
    this.notFound = new Set();
    this.headerFooterLinks = new Set();
    this.linkSources = new Map(); // url -> [referrer1, referrer2]
    this.robotsCache = {}; // To persist robots.txt data
    this.load();
  }

  load() {
    if (fs.existsSync(this.stateFile)) {
      try {
        const state = fs.readJSONSync(this.stateFile);
        this.visited = new Set(state.visited || []);
        this.toVisit = new Map(state.toVisit || []);
        this.failed = new Map(state.failed || []);
        this.content = new Map(state.content || []);
        this.notFound = new Set(state.notFound || []);
        this.headerFooterLinks = new Set(state.headerFooterLinks || []);
        this.linkSources = new Map(state.linkSources || []);
        this.robotsCache = state.robotsCache || {};
        robotsCache = this.robotsCache; // Update global cache
        console.log(`Loaded state: ${this.visited.size} visited, ${this.toVisit.size} to visit, ${this.notFound.size} 404s`);
      } catch (error) {
        console.error('Error loading state:', error);
        this.robotsCache = {}; // Initialize if error
      }
    }
  }

  save() {
    const state = {
      visited: Array.from(this.visited),
      toVisit: Array.from(this.toVisit),
      failed: Array.from(this.failed),
      content: Array.from(this.content),
      notFound: Array.from(this.notFound),
      headerFooterLinks: Array.from(this.headerFooterLinks),
      linkSources: Array.from(this.linkSources),
      robotsCache: this.robotsCache,
      lastUpdated: new Date().toISOString()
    };
    try {
      fs.writeJSONSync(this.stateFile, state, { spaces: 2 });
    } catch (error) {
      console.error('Error saving state:', error);
    }
  }

  addUrl(url, referrer = null, currentDepth = 0) {
    const normalized = normalizeUrl(url);
    if (!normalized) return;

    if (this.visited.has(normalized) || this.toVisit.has(normalized) || this.notFound.has(normalized)) {
      return;
    }
    this.toVisit.set(normalized, { referrer, addedAt: new Date().toISOString(), depth: currentDepth });

    if (referrer) {
      const sources = this.linkSources.get(normalized) || [];
      sources.push(referrer);
      this.linkSources.set(normalized, sources);
    }
  }

  getNextUrl() {
    if (this.toVisit.size === 0) return null;
    const [url, metadata] = this.toVisit.entries().next().value;
    this.toVisit.delete(url);
    return { url, metadata };
  }

  markVisited(url) {
    this.visited.add(normalizeUrl(url));
  }

  markFailed(url, error, referrer = null) {
    const normalized = normalizeUrl(url);
    this.failed.set(normalized, {
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      referrer: referrer
    });
  }

  mark404(url, referrer = null) {
    const normalized = normalizeUrl(url);
    this.notFound.add(normalized);
    console.log(`404 marked: ${url} (referred by: ${referrer || 'unknown'})`);
    const notFoundLog = path.join(config.outputDir, '404_pages.log');
    fs.appendFileSync(notFoundLog, `[${new Date().toISOString()}] 404: ${url} <- ${referrer || 'direct'}\n`);
  }

  markAsHeaderFooterLink(url) {
    this.headerFooterLinks.add(normalizeUrl(url));
  }
  isHeaderFooterLink(url) {
    return this.headerFooterLinks.has(normalizeUrl(url));
  }
  saveContent(url, contentData) {
    this.content.set(normalizeUrl(url), contentData);
  }
}

// Authentication handler
async function authenticateEPFL(page, currentAttemptUrl) {
  console.log('Starting EPFL authentication process...');

  try {
    // First, navigate to a known protected page if not already on a login sequence
    // to reliably trigger the authentication flow.
    if (!page.url().includes('login') && !page.url().includes('tequila') && !page.url().includes('microsoftonline')) {
        console.log(`Not on a login page (current: ${page.url()}). Navigating to inside.epfl.ch to trigger login...`);
        await page.goto('https://inside.epfl.ch', { waitUntil: 'networkidle2', timeout: 45000 });
        await waitRandomTime(config.delays.navigation); // Wait for potential redirects
    }

    // Check if already logged in after navigation attempt or if already on a relevant page
    const isAlreadyLoggedIn = await page.evaluate(() =>
      document.body.innerText.toLowerCase().includes('logout') ||
      document.body.innerText.toLowerCase().includes('déconnexion') ||
      document.querySelector('.user-logged-in, [class*="logged-in"], [data-test-id="logout-button"]')
    );

    if (isAlreadyLoggedIn) {
      console.log('Already authenticated or no immediate redirect to login detected.');
      if (normalizeUrl(page.url()) !== normalizeUrl(currentAttemptUrl)) {
        console.log(`Authentication check done. Current URL: ${page.url()}. Redirecting to original target: ${currentAttemptUrl}`);
        await page.goto(currentAttemptUrl, { waitUntil: 'networkidle2', timeout: 60000 });
      }
      return;
    }
    console.log('Current URL after initial nav/check, proceeding with authentication:', page.url());

    // TEQUILA LOGIN
    if (page.url().includes('tequila.epfl.ch')) {
      console.log('Detected Tequila login form. Using Tequila-specific username.');
      await page.waitForSelector('#username', { timeout: 15000, visible: true });
      await page.type('#username', config.credentials.usernameTequila, { delay: config.delays.typing });
      await page.type('#password', config.credentials.password, { delay: config.delays.typing });
      await waitRandomTime(500);

      // Prioritized selectors: #loginbutton first.
      const tequilaLoginSelectors = [
        '#loginbutton', // Primary, try this first with more patience
        'input[type="image"][name="login"]', // The image button variant
        'button[type="submit"]', // Generic submit button
        'input[type="submit"][name="login"]',
        'input[type="submit"][value*="Login" i]', // Case-insensitive "Login"
        'input[type="submit"][value*="Connecter" i]' // Case-insensitive "Se Connecter" / "Connecter"
      ];
      let tequilaLoginClicked = false;

      for (const selector of tequilaLoginSelectors) {
          try {
              const selectorTimeout = (selector === '#loginbutton') ? 5000 : 3000; // More time for primary selector
              await page.waitForSelector(selector, { timeout: selectorTimeout, visible: true });
              console.log(`Attempting to click Tequila login button with selector: ${selector}`);

              await Promise.all([
                  page.waitForNavigation({ waitUntil: 'load', timeout: 60000 }), // Changed to 'load'
                  page.click(selector)
              ]);

              tequilaLoginClicked = true;
              console.log(`Clicked Tequila login button with selector: ${selector}. Waiting for post-load scripts...`);
              await setTimeout(1500); // Give time for scripts on the new page to run after 'load' event
              break;
          } catch (e) {
              console.log(`Tequila login selector "${selector}" failed. Error: ${e.message.split('\n')[0]}. Trying next.`);
          }
      }

      if (!tequilaLoginClicked) {
          throw new Error('Could not find or click any of the Tequila login buttons after trying all known selectors.');
      }
      console.log('Submitted Tequila credentials. Current URL after Tequila submit:', page.url());
    }

    // MICROSOFT LOGIN (Can follow Tequila or be direct)
    // Check again as Tequila might redirect here
    if (page.url().includes('login.microsoftonline.com')) {
      console.log('Detected Microsoft login form. Using Microsoft-specific username.');

      if (await page.$('#i0116')) {
        try {
          console.log('Entering email for Microsoft login...');
          await page.waitForSelector('#i0116', { timeout: 20000, visible: true });
          await page.type('#i0116', config.credentials.usernameMicrosoft + "@epfl.ch", { delay: config.delays.typing });
          await waitRandomTime(500);
          await Promise.all([
            page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 60000 }),
            page.click('#idSIButton9')
          ]);
          console.log('Submitted Microsoft email. Current URL:', page.url());
        } catch (e) {
          console.log(`Microsoft email field interaction failed: ${e.message.split('\n')[0]}. Might be pre-filled or flow changed.`);
        }
      } else {
        console.log('Microsoft email field (#i0116) not immediately found. Assuming it might be pre-filled or on a different step.');
      }

      if (await page.$('#i0118')) {
        try {
          console.log('Entering password for Microsoft login...');
          await page.waitForSelector('#i0118', { timeout: 20000, visible: true });
          await page.type('#i0118', config.credentials.password, { delay: config.delays.typing });
          await waitRandomTime(500);
          await Promise.all([
            page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 60000 }),
            page.click('#idSIButton9')
          ]);
          console.log('Submitted Microsoft password. Current URL:', page.url());
        } catch (e) {
          console.log(`Microsoft password field interaction failed: ${e.message.split('\n')[0]}.`);
        }
      } else {
         console.log('Microsoft password field (#i0118) not immediately found.');
      }

      try {
        console.log('Checking for Microsoft MFA prompt (text-based indicators)...');
        await page.waitForFunction(() => {
            const textIndicators = ['verify your identity', 'approuver la demande', 'approve the request', 'enter the number', 'entrez le code', 'mfa', 'authenticator'];
            const pageText = document.body.innerText.toLowerCase();
            return textIndicators.some(text => pageText.includes(text));
        }, { timeout: 20000 });

        console.log('MFA prompt indicators found. Attempting to extract challenge number...');
        const mfaNumberSelectors = ['.text-title', '[data-testid="ChallengeTitle"]', '.displaySign', 'div[role="heading"][aria-level="1"]'];
        let mfaNumberFound = null;
        for (const selector of mfaNumberSelectors) {
            const element = await page.$(selector);
            if (element) {
                const text = await element.evaluate(el => el.innerText.trim());
                const match = text.match(/\d{2,}/);
                if (match && match[0]) {
                    mfaNumberFound = match[0];
                    console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
                    console.log(`!! MFA Action Required: Enter the number "${mfaNumberFound}" in your Microsoft Authenticator app.`);
                    console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
                    break;
                }
            }
        }
        if (!mfaNumberFound) {
            console.log("MFA challenge detected, but couldn't extract the specific number with current selectors, OR it's an 'approve' type notification. Please check your Authenticator app for a prompt.");
        }

        console.log('Waiting for MFA completion (up to 2 minutes)... Current URL:', page.url());
        await page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 120000 });
        console.log('MFA seemingly completed. Current URL after MFA:', page.url());

      } catch (e) {
        console.log(`No specific MFA number challenge detected or MFA step timed out/navigated away: ${e.message.split('\n')[0]}. It might have been a different MFA type or completed quickly.`);
      }

      if (page.url().includes('login.microsoftonline.com') || page.url().includes('kmsi')) {
        try {
          console.log('Checking for "Stay signed in?" prompt...');
          const staySignedInButton = await page.waitForSelector('#idSIButton9, #acceptButton, input[type="submit"][value*="Yes" i], button::-p-text(Yes)', { timeout: 15000, visible: true });
          if (staySignedInButton) {
            await waitRandomTime(500);
            await Promise.all([
              page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 60000 }),
              staySignedInButton.click()
            ]);
            console.log('Handled "Stay signed in?" prompt. Current URL:', page.url());
          } else {
            console.log('"Stay signed in?" button not found with current selectors.');
          }
        } catch (e) {
          console.log(`"Stay signed in?" prompt not found or interaction failed: ${e.message.split('\n')[0]}.`);
        }
      }
    }

    console.log('Verifying redirection to an EPFL domain and not a login page...');
    await page.waitForFunction(
      () => { // Removed unused parameter targetUrlAfterLogin
        const currentHref = window.location.href;
        const onEpflDomain = currentHref.includes('.epfl.ch');
        const notOnLoginPages = !currentHref.includes('login.') &&
                                !currentHref.includes('tequila.epfl.ch') &&
                                !currentHref.includes('microsoftonline.com') &&
                                !currentHref.includes('consent') &&
                                !currentHref.includes('auth');
        return (onEpflDomain && notOnLoginPages);
      },
      { timeout: 60000 }
    );
    console.log('Authentication flow finished. Current URL:', page.url());

    if (normalizeUrl(page.url()) !== normalizeUrl(currentAttemptUrl)) {
        console.log(`Authentication successful, but not on original target. Navigating from ${page.url()} to ${currentAttemptUrl}`);
        await page.goto(currentAttemptUrl, { waitUntil: 'networkidle2', timeout: 60000 });
        console.log('Final navigation to original target page. Current URL:', page.url());
    } else {
        console.log('Authentication successful and on the target page or a valid post-login EPFL page.');
    }

  } catch (error) {
    console.error(`Authentication failed. Current URL at failure: ${page.url()}. Error: ${error.message}`);
    const screenshotPath = path.join(config.outputDir, `auth_error_${Date.now()}.png`);
    try {
      if (!page.isClosed()) {
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`Authentication error screenshot saved to: ${screenshotPath}`);
      } else {
        console.log('Page was closed, cannot take screenshot for authentication error.');
      }
    } catch (ssError) {
      console.error(`Failed to take screenshot during auth error: ${ssError.message}`);
    }
    throw error;
  }
}

async function expandAllCollapsedContent(page) {
  console.log('Expanding collapsed content within main content areas...');
  // Combine main and article selectors to define the scope for expansion
  const mainContentAreaQuery = [
    ...config.contentExtraction.selectors.main,
    ...config.contentExtraction.selectors.article
  ].join(', ');

  // Selectors for elements that trigger collapse/expand, to be searched *within* main content
  const collapsibleTriggerSelectors = [
    'button[data-toggle="collapse"]', // Matches .collapsed and :not(.collapsed)
    'a[data-toggle="collapse"]',
    '.accordion-toggle', '.accordion-button', // Common accordion classes
    'details:not([open]) > summary', // Specifically target summary of unopened details
    '.collapse-title.collapsed', '.expandable-header.collapsed',
    '.toggle-content:not(.expanded) > *:first-child', // Click the first child of a toggle
    '[aria-expanded="false"]', // Generic attribute selector
    '.show-more-button', '.expand-button', '.read-more',
    // More general patterns, be cautious if they are too broad
    // '[class*="collapsible"] button', '[class*="accordion"] button',
  ];

  let totalExpandedCount = 0;

  // Iterate a few times to catch nested or dynamically revealed collapsible elements
  for (let pass = 0; pass < 3; pass++) { // Max 3 passes
    let expandedInThisPass = await page.evaluate((mainContainersQueryString, childTriggerQueries) => {
      let currentPassExpansionCount = 0;
      const mainContentElements = Array.from(document.querySelectorAll(mainContainersQueryString));

      if (mainContentElements.length === 0) {
        // If your main/article selectors don't find anything, no targeted expansion can occur.
        // Consider if a fallback is needed or if this is acceptable.
        // For now, it means no elements will be sought for expansion.
        return 0;
      }

      for (const mainElement of mainContentElements) {
        for (const triggerQuery of childTriggerQueries) {
          const triggerElements = Array.from(mainElement.querySelectorAll(triggerQuery));
          for (const element of triggerElements) {
            try {
              // Check if element is visible and not already expanded
              const style = window.getComputedStyle(element);
              const isVisible = style.display !== 'none' && style.visibility !== 'hidden' && element.getClientRects().length > 0;

              let isAlreadyExpanded = element.getAttribute('aria-expanded') === 'true';
              if (element.tagName === 'SUMMARY' && element.parentElement.tagName === 'DETAILS') {
                isAlreadyExpanded = element.parentElement.hasAttribute('open');
              }
              // For data-toggle, check if it has 'collapsed' class (if applicable to your targets)
              if (element.hasAttribute('data-toggle') && element.classList.contains('collapsed') === false && !isAlreadyExpanded) {
                // If it's a data-toggle element and NOT collapsed, it might be considered expanded for some definitions
                // This part might need adjustment based on how "collapsed" vs "expanded" is specifically marked.
                // For now, primary check is aria-expanded and details[open].
              }


              if (isVisible && !isAlreadyExpanded) {
                if (typeof element.click === 'function') {
                  element.click(); // Click the element
                  currentPassExpansionCount++;
                  // Note: Adding async delays inside evaluate is complex.
                  // The delay between passes in Node.js handles waiting for content to load.
                }
              }
            } catch (e) {
              // Silently ignore errors for individual clicks (e.g., element became stale)
              // console.warn(`Could not click or evaluate element for expansion: ${e.message}`);
            }
          }
        }
      }
      return currentPassExpansionCount;
    }, mainContentAreaQuery, collapsibleTriggerSelectors);

    if (expandedInThisPass > 0) {
      totalExpandedCount += expandedInThisPass;
      console.log(`  Expansion Pass ${pass + 1}: Expanded ${expandedInThisPass} elements within main content. Waiting for dynamic content...`);
      await waitRandomTime(800 + (pass * 200)); // Wait a bit longer after each pass
    } else {
      console.log(`  Expansion Pass ${pass + 1}: No new elements expanded within main content.`);
      break; // Exit if no elements were expanded in this pass
    }
  }

  if (totalExpandedCount > 0) {
    console.log(`Finished expanding content: ${totalExpandedCount} elements expanded in total within main content areas.`);
  } else {
    console.log('No collapsible content found or expanded within main content areas.');
  }
}

async function extractPageContent(page, currentUrl) {
  return await page.evaluate((selectorsToRemove, mainContentSelectors, articleSelectors, url) => {
    // Clone the body to avoid modifying the live DOM if other operations need it later
    const docClone = document.documentElement.cloneNode(true);

    // Remove unwanted elements from the clone
    docClone.querySelectorAll(selectorsToRemove.join(', ')).forEach(el => el.remove());

    const findTextInClone = (targetSelectors) => {
      let contentText = '';
      for (const selector of targetSelectors) {
        const elements = docClone.querySelectorAll(selector);
        if (elements.length > 0) {
          elements.forEach(el => { contentText += el.innerText.trim() + '\n\n'; });
          return contentText.trim(); // Found content in primary selectors
        }
      }
      return null;
    };

    let mainText = findTextInClone(mainContentSelectors) || findTextInClone(articleSelectors);

    if (!mainText) { // Fallback to body if specific content selectors yield nothing
        const bodyEl = docClone.querySelector('body');
        mainText = bodyEl ? bodyEl.innerText.trim() : '';
    }

    // Extract links from the original document to ensure all links are found
    const allLinks = Array.from(document.querySelectorAll('a[href]'))
      .map(a => {
        try {
          return {
            href: new URL(a.getAttribute('href'), document.baseURI).toString(), // Ensure absolute URL
            text: a.innerText.trim().replace(/\s+/g, ' ') || '', // Normalize spaces
            // Basic categorization attempt (can be refined)
            isNavigation: ['header', 'nav', 'footer', '.navbar', '.nav-menu', 'aside', '.sidebar'].some(sel => a.closest(sel) !== null)
          };
        } catch (e) { return null; } // Invalid href
      })
      .filter(link => link && link.href && !link.href.startsWith('javascript:') && !link.href.startsWith('mailto:') && !link.href.startsWith('tel:'));


    const title = document.title || '';
    const metaDescriptionTag = document.querySelector('meta[name="description"]');
    const metaKeywordsTag = document.querySelector('meta[name="keywords"]');

    const metadata = {
      title: title,
      description: metaDescriptionTag ? metaDescriptionTag.content : '',
      keywords: metaKeywordsTag ? metaKeywordsTag.content : '',
      language: document.documentElement.lang || 'en',
      lastModified: document.lastModified || '',
      url: url
    };
    
    // 404 check based on title/content from original document
    const is404 = title.toLowerCase().includes('not found') || title.includes('404') ||
                  document.body.innerText.toLowerCase().includes('page not found') ||
                  document.body.innerText.toLowerCase().includes("doesn't exist");

    return {
      title: title,
      text: mainText,
      links: allLinks,
      metadata: metadata,
      is404: is404
    };
  }, config.contentExtraction.selectors.remove, config.contentExtraction.selectors.main, config.contentExtraction.selectors.article, currentUrl);
}


async function savePageData(url, contentData, outputDir) {
  const fileName = generateFileNameFromUrl(url);

  const dataToSave = {
    url: url,
    crawledAt: new Date().toISOString(),
    title: contentData.title,
    textLength: contentData.text.length,
    extractedText: contentData.text, // For RAG
    linksCount: contentData.links.length,
    // links: contentData.links, // Optional: save all links
    metadata: contentData.metadata,
  };

  const jsonPath = path.join(outputDir, 'json', `${fileName}.json`);
  fs.ensureDirSync(path.dirname(jsonPath));
  fs.writeJSONSync(jsonPath, dataToSave, { spaces: 2 });

  const textPath = path.join(outputDir, 'text', `${fileName}.txt`);
  fs.ensureDirSync(path.dirname(textPath));
  fs.writeFileSync(textPath, `URL: ${url}\nTitle: ${contentData.title}\n\n${contentData.text}`);

  const mdPath = path.join(outputDir, 'markdown', `${fileName}.md`);
  fs.ensureDirSync(path.dirname(mdPath));
  const mdContent = `# ${contentData.title}\n\nSource: <${url}>\nFetched: ${dataToSave.crawledAt}\n\n${contentData.text}`;
  fs.writeFileSync(mdPath, mdContent);
}

async function downloadFile(page, url, outputDir) {
  const urlObj = new URL(url);
  const fileExtension = path.extname(urlObj.pathname) || '.file';
  const fileName = generateFileNameFromUrl(url) + fileExtension;
  const downloadsDir = path.join(outputDir, 'downloads');
  fs.ensureDirSync(downloadsDir);
  const filePath = path.join(downloadsDir, fileName);

  console.log(`Attempting to download (Node HTTPS method): ${url} to ${filePath}`);
  let success = false;

  try {
    const cookiesArray = await page.cookies(url); // Get cookies from the Puppeteer page context
    const cookieHeader = cookiesArray.map(cookie => `${cookie.name}=${cookie.value}`).join('; ');

    const options = {
      hostname: urlObj.hostname,
      path: urlObj.pathname + urlObj.search,
      method: 'GET',
      headers: {
        'User-Agent': getRandomUserAgent(), // Use one of your UAs
        'Cookie': cookieHeader,
        // 'Referer': page.url() // Optional: sometimes helps, use the URL of the page containing the link
      },
      rejectUnauthorized: process.env.NODE_ENV !== 'development', // Handles self-signed certs in dev if needed
    };

    const requester = urlObj.protocol === 'https:' ? https : http;

    await new Promise((resolve, reject) => {
      const request = requester.get(options, (response) => {
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download ${url}. Status Code: ${response.statusCode} ${response.statusMessage}`));
          response.resume(); // consume response data to free up memory
          return;
        }

        const contentType = response.headers['content-type'];
        console.log(`[Node-HTTPS] Content-Type for ${url}: ${contentType}`);

        if (!contentType || (!contentType.includes('application/pdf') && !contentType.includes('application/octet-stream'))) {
            console.error(`[Node-HTTPS] Error: Expected PDF content-type, but got ${contentType} for ${url}`);
            const errorFilePath = filePath.replace(/\.[^/.]+$/, `_error_content_nodejs.dat`);
            const errorFileStream = fs.createWriteStream(errorFilePath);
            response.pipe(errorFileStream);
            errorFileStream.on('finish', () => {
                console.error(`[Node-HTTPS] Non-PDF content saved to ${errorFilePath} for inspection.`);
                reject(new Error(`Expected PDF content-type, got ${contentType}`));
            });
            return;
        }

        const fileStream = fs.createWriteStream(filePath);
        response.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close(() => { // close() is async, call resolve in its callback
            console.log(`[Node-HTTPS] File downloaded and stream closed: ${fileName}`);
            success = true;
            resolve();
          });
        });

        fileStream.on('error', (err) => {
          fs.unlink(filePath, () => {}); // Delete the potentially corrupted file
          reject(new Error(`Error writing file stream for ${url}: ${err.message}`));
        });
      });

      request.on('error', (err) => {
        reject(new Error(`Error with HTTP(S) request for ${url}: ${err.message}`));
      });

      // Set a timeout for the request itself
      request.setTimeout(config.delays.download * 2, () => {
        request.destroy(); // Abort the request
        reject(new Error(`Request timed out for ${url} after ${config.delays.download * 2 / 1000}s`));
      });

      request.end();
    });

    if (success && fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      if (stats.size > 0) {
        console.log(`File saved successfully (Node HTTPS method): ${fileName} (${(stats.size / 1024).toFixed(2)} KB)`);
        // Check magic number for PDF
        const fileBuffer = fs.readFileSync(filePath, { length: 4 }); // Read only first 4 bytes
        if (fileBuffer.toString('ascii', 0, 4) !== '%PDF') {
            console.warn(`[Node-HTTPS] Downloaded file ${fileName} does not start with PDF magic number.`);
            // Optionally rename/delete if strict checking is required
            // For now, we'll assume if server sent it as PDF and it has size, it's okay.
        }
      } else {
        console.error(`Downloaded file (Node HTTPS method) is empty: ${fileName}`);
        fs.unlinkSync(filePath); // Clean up empty file
        success = false;
      }
    } else if (success) { // success was true but file doesn't exist (should not happen if stream finished)
        console.error(`File stream finished for ${fileName}, but file not found or invalid.`);
        success = false;
    }


  } catch (error) {
    console.error(`Error during downloadFile (Node HTTPS method) for ${url}: ${error.message}`);
    success = false;
  }

  if (success) {
    return filePath;
  } else {
    // Attempt to clean up if a corrupted/empty file was created but not deemed a success
    if (fs.existsSync(filePath)) {
      try {
        if (fs.statSync(filePath).size === 0 || !success) fs.unlinkSync(filePath);
      } catch (e) { /* ignore */ }
    }
    return null;
  }
}

async function crawlPage(browser, state, urlInfo, currentDepth = 0) {
  const { url, metadata } = urlInfo;
  const referrer = metadata ? metadata.referrer : null;

  if (currentDepth > config.crawling.maxDepth) {
    console.log(`Skipping ${url} - max depth exceeded.`);
    return;
  }
  if (state.visited.size >= config.crawling.maxPages) {
    console.log('Max pages reached. Stopping crawl.');
    return; // This will eventually stop the pool
  }

  if (!await isAllowedByRobots(url)) {
    console.log(`Skipping (disallowed by robots.txt): ${url}`);
    state.markFailed(url, new Error('Disallowed by robots.txt'), referrer); // Or a different state
    return;
  }

  const page = await browser.newPage();
  await page.setUserAgent(getRandomUserAgent());
  await page.setDefaultNavigationTimeout(60000); // 60s
  await page.setDefaultTimeout(45000); // 45s

  try {
    console.log(`Crawling [D:${currentDepth}, V:${state.visited.size}]: ${url} (from: ${referrer || 'seed'})`);

    const response = await page.goto(url, { waitUntil: 'networkidle2' });

    if (!response) {
        throw new Error('Navigation returned no response.');
    }

    // HTTP Status Check
    const status = response.status();
    if (status === 404) {
      console.log(`HTTP 404: ${url}`);
      state.mark404(url, referrer);
      return;
    } else if (status >= 400 && status <= 599) {
      console.error(`HTTP Error ${status}: ${url}`);
      state.markFailed(url, new Error(`HTTP ${status} ${response.statusText()}`), referrer);
      return;
    }
    
    // Authentication check & handling
    // It will try to navigate to inside.epfl.ch if not on a login page, then back to original URL
    const needsAuth = page.url().includes('login.') || page.url().includes('tequila') || page.url().includes('microsoftonline');
    if (needsAuth || response.url().includes('login.') || response.url().includes('tequila') || response.url().includes('microsoftonline')) {
      await authenticateEPFL(page, url); // Pass original target URL
      // It's assumed authenticateEPFL navigates back to the original URL or an equivalent page.
      // If not, the page.url() might be different here.
    }


    await waitRandomTime(config.delays.navigation / 2); // Shorter delay, more randomness

    // Check if URL is a downloadable file type based on its extension
    const urlPath = new URL(url).pathname.toLowerCase();
    const isDownloadableFileType = config.crawling.includeFileTypes.some(ext => urlPath.endsWith(ext));

    if (isDownloadableFileType) {
        console.log(`Processing as downloadable file: ${url}`);
        await downloadFile(page, url, config.outputDir);
        // Even for downloads, we mark as visited to avoid re-processing the URL itself
        state.markVisited(url);
        state.save(); // Save state after processing a downloadable
        return; // Stop further processing for this URL
    }

    // For HTML pages:
    await expandAllCollapsedContent(page);
    const contentData = await extractPageContent(page, url);

    if (contentData.is404) {
      console.log(`Content-based 404 detected: ${url}`);
      state.mark404(url, referrer);
      return;
    }

    await savePageData(url, contentData, config.outputDir);
    state.saveContent(url, { ...contentData, crawledAt: new Date().toISOString() }); // Save to CrawlerState.content

    state.markVisited(url);
    pagesSinceLastBrowserLaunch++;

    // Process and add new links
    if (currentDepth < config.crawling.maxDepth) {
      let addedLinks = 0;
      for (const linkObj of contentData.links) {
        const absoluteLink = normalizeUrl(new URL(linkObj.href, url).toString());
        if (absoluteLink && isAllowedUrl(absoluteLink, url) && !state.isHeaderFooterLink(absoluteLink)) {
          state.addUrl(absoluteLink, url, currentDepth + 1);
          addedLinks++;
          // Simple heuristic to identify common nav links
          if (linkObj.isNavigation) {
            const sources = state.linkSources.get(absoluteLink) || [];
            if (sources.length > 5) { // Mark as common nav after seeing it from 5+ different pages
                state.markAsHeaderFooterLink(absoluteLink);
            }
          }
        }
      }
      console.log(`  Found ${contentData.links.length} links, added ${addedLinks} new unique & allowed URLs to queue.`);
    }
    state.save(); // Save state more frequently
  } catch (error) {
    console.error(`Error crawling ${url}:`, error.message, error.stack);
    state.markFailed(url, error, referrer);
    state.save();
  } finally {
    try {
      await page.goto('about:blank'); // Help free up resources
      await page.close();
    } catch (closeError) {
      console.error(`Error closing page for ${url}:`, closeError.message);
    }
  }
}

class CrawlerPool {
  constructor(state, concurrency) {
    this.state = state;
    this.concurrency = concurrency;
    this.activeCrawlers = 0;
    this.browser = null; // Browser managed by the pool
    this.shouldStop = false;
  }

  async launchBrowser() {
      console.log("Launching new browser instance...");
      if (this.browser) {
          try { await this.browser.close(); } catch (e) { console.error("Error closing old browser:", e); }
      }
      this.browser = await puppeteer.launch({
        headless: true, // Set to false for debugging auth
        args: [
          '--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage',
          '--disable-accelerated-2d-canvas', '--disable-gpu',
          '--window-size=1920x1080',
        ],
        defaultViewport: { width: 1920, height: 1080 },

        protocolTimeout: 60000

      });
      this.browser.on('disconnected', () => {
          console.warn('Browser disconnected. Attempting to relaunch in next cycle if needed.');
          this.browser = null; // Mark browser as null so it gets relaunched
      });
      lastBrowserLaunchTime = Date.now();
      pagesSinceLastBrowserLaunch = 0;
      return this.browser;
  }

  async needsBrowserRestart() {
      if (!this.browser || !this.browser.isConnected()) return true;
      if (pagesSinceLastBrowserLaunch >= config.crawling.relaunchBrowserAfterPages) return true;
      const hoursSinceLastLaunch = (Date.now() - lastBrowserLaunchTime) / (1000 * 60 * 60);
      if (hoursSinceLastLaunch >= config.crawling.relaunchBrowserAfterHours) return true;
      return false;
  }

  async run() {
    this.shouldStop = false;
    if (!this.browser || !(await this.browser.isConnected()) || await this.needsBrowserRestart()) {
        await this.launchBrowser();
    }

    const processNext = async () => {
        if (this.shouldStop) {
            this.activeCrawlers--;
            if (this.activeCrawlers === 0) this.finalizeRun();
            return;
        }

        if (await this.needsBrowserRestart() && this.activeCrawlers === 0) {
            await this.launchBrowser();
        }
        // Ensure browser is connected before grabbing next URL
        if (!this.browser || !this.browser.isConnected()) {
            console.log("Browser not available or disconnected, attempting relaunch or waiting.");
            if (this.activeCrawlers === 0) await this.launchBrowser(); // Relaunch if no active crawlers
            else { // wait for active crawlers to finish before relaunching
                await setTimeout(5000); 
                this.activeCrawlers--; // Decrement as if one finished, to re-trigger loop condition
                if (this.activeCrawlers < this.concurrency) processQueue(); // try to fill slot again
                return;
            }
        }


        if (this.state.visited.size >= config.crawling.maxPages) {
            console.log('Max pages limit reached. Stopping crawl tasks.');
            this.shouldStop = true;
            this.activeCrawlers--;
            if (this.activeCrawlers === 0) this.finalizeRun();
            return;
        }

        const urlInfo = this.state.getNextUrl();
        if (urlInfo) {
            await crawlPage(this.browser, this.state, urlInfo, urlInfo.metadata.depth)
                .catch(e => console.error(`Unhandled error in crawlPage for ${urlInfo.url}:`, e))
                .finally(() => {
                    this.activeCrawlers--;
                    if (!this.shouldStop && this.activeCrawlers < this.concurrency) {
                        processQueue(); // Try to fill the empty slot
                    } else if (this.activeCrawlers === 0 && (this.shouldStop || this.state.toVisit.size === 0)) {
                        this.finalizeRun();
                    }
                });
        } else { // No more URLs
            this.activeCrawlers--;
            if (this.activeCrawlers === 0) this.finalizeRun();
        }
    };
    
    const processQueue = () => {
        while (this.activeCrawlers < this.concurrency && this.state.toVisit.size > 0 && !this.shouldStop) {
            if (this.state.visited.size >= config.crawling.maxPages) {
                 this.shouldStop = true;
                 break;
            }
            this.activeCrawlers++;
            processNext();
        }
        if (this.activeCrawlers === 0 && (this.shouldStop || this.state.toVisit.size === 0)) {
            this.finalizeRun();
        }
    };
    
    this.finalizeRun = () => {
        if (!this.finalizing) {
            this.finalizing = true; // Prevent multiple calls
            console.log("All crawl tasks finished or pool stopped.");
            if (this.onFinishedCallback) this.onFinishedCallback();
        }
    };

    // Initial fill of the queue
    processQueue();

    // Return a promise that resolves when crawling is done
    return new Promise(resolve => {
        this.onFinishedCallback = resolve;
        // Safety check if queue is empty initially or stopping criteria met
        if (this.activeCrawlers === 0 && (this.shouldStop || this.state.toVisit.size === 0)) {
             this.finalizeRun();
        }
    });
  }

  async stop() {
    console.log("Attempting to gracefully stop the crawler pool...");
    this.shouldStop = true;
    // It will stop picking new tasks. Existing tasks will complete.
  }
}


async function main() {
  console.log('Starting EPFL full site scraper for RAG...');
  console.log(`Output directory: ${config.outputDir}`);

  fs.ensureDirSync(config.outputDir);
  ['json', 'text', 'markdown', 'downloads'].forEach(subDir => {
    fs.ensureDirSync(path.join(config.outputDir, subDir));
  });

  const stateFile = path.join(config.outputDir, 'crawler_state.json');
  const state = new CrawlerState(stateFile);

  if (state.toVisit.size === 0 && state.visited.size === 0) {
    config.baseUrls.forEach(url => state.addUrl(url, null, 0)); // Add with depth 0
    state.save();
  }

  const logFile = path.join(config.outputDir, 'crawler.log');
  const originalConsoleLog = console.log;
  const originalConsoleError = console.error;

  const logToFile = (level, ...args) => {
    const message = args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg)).join(' ');
    const logMessage = `[${new Date().toISOString()}] [${level.toUpperCase()}] ${message}\n`;
    fs.appendFileSync(logFile, logMessage);
    if (level === 'error') originalConsoleError.apply(console, args);
    else originalConsoleLog.apply(console, args);
  };

  console.log = (...args) => logToFile('info', ...args);
  console.error = (...args) => logToFile('error', ...args);
  console.warn = (...args) => logToFile('warn', ...args);


  console.log('Crawler started. Initial queue size:', state.toVisit.size);

  const pool = new CrawlerPool(state, config.crawling.concurrentPages);
  
  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log("SIGINT received. Shutting down crawler gracefully...");
    await pool.stop();
    // Wait a bit for active tasks to finish, then force exit if necessary
    setTimeout(async () => {
        if (pool.browser && pool.browser.isConnected()) {
            await pool.browser.close();
        }
        console.log("Exiting now.");
        process.exit(0);
    }, 30000); // Give 30 seconds for tasks to attempt to finish
  });


  await pool.run(); // This now returns a promise that resolves when done

  console.log('Crawler pool finished.');

  if (pool.browser && pool.browser.isConnected()) {
    await pool.browser.close();
  }
  
  state.save(); // Final save

  console.log(`Crawling completed. Total pages visited: ${state.visited.size}, Failed: ${state.failed.size}, 404s: ${state.notFound.size}`);

  // Generate summary report (simplified)
  const report = {
    totalPagesVisited: state.visited.size,
    failedPages: state.failed.size,
    notFoundPages: state.notFound.size,
    contentItems: state.content.size,
    crawlEndTime: new Date().toISOString(),
  };
  fs.writeJSONSync(path.join(config.outputDir, 'crawl_summary_report.json'), report, { spaces: 2 });
  console.log('Summary report generated.');
}

main().catch(error => {
  console.error('Unhandled error in main:', error);
  process.exit(1);
});