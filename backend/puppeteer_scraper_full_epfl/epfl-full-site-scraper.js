// epfl-full-site-scraper.js
const puppeteer = require('puppeteer');
const fs = require('fs-extra');
const path = require('path');
const dotenv = require('dotenv');
const { setTimeout } = require('timers/promises');
const crypto = require('crypto');
const { URL } = require('url');

// Load environment variables
const envPath = path.resolve(__dirname, '../.env');
dotenv.config({ path: envPath });

// Check credentials
if (!process.env.EPFL_USERNAME || !process.env.EPFL_PASSWORD) {
  console.error('Error: EPFL_USERNAME and EPFL_PASSWORD must be defined in .env file');
  process.exit(1);
}

// Configuration
const config = {
  baseUrls: [
    'https://inside.epfl.ch',
    'https://www.epfl.ch',
    'https://support.epfl.ch',
    'https://wiki.epfl.ch',
    'https://memento.epfl.ch',
  ],
  outputDir: path.join(__dirname, 'epfl_full_site_data'),
  credentials: {
    username: process.env.EPFL_USERNAME,
    password: process.env.EPFL_PASSWORD
  },
  crawling: {
    maxDepth: 5,                    // Maximum crawl depth
    maxPages: 10000,                // Maximum pages to crawl
    concurrentPages: 3,             // Concurrent browser pages
    respectRobotsTxt: true,
    allowedDomains: [
      'epfl.ch',
      'support.epfl.ch',
      'wiki.epfl.ch',
      'memento.epfl.ch',
      'inside.epfl.ch'
    ],
    excludePatterns: [
      /\.(jpg|jpeg|png|gif|svg|ico|webp|bmp)$/i,
      /\.(mp4|avi|mov|wmv|flv|webm)$/i,
      /\.(mp3|wav|ogg|flac)$/i,
      /\.(zip|rar|7z|tar|gz)$/i,
      /\/logout/i,
      /\/signout/i,
      /\/deconnexion/i
    ],
    includeFileTypes: [
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
    typing: 50,
    download: 2000,
    randomMin: 300,
    randomMax: 1500,
    rateLimitDelay: 1000  // Delay between requests to same domain
  },
  userAgents: [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
  ],
  contentExtraction: {
    selectors: {
      // Common content selectors for EPFL sites
      main: ['main', '#main', '.main-content', '#content', '.content'],
      article: ['article', '.article', '.post', '.entry-content'],
      navigation: ['nav', '.navigation', '#navigation'],
      sidebar: ['aside', '.sidebar', '#sidebar'],
      // Remove these elements
      remove: ['script', 'style', 'noscript', 'iframe', '.advertisement', '#cookie-banner', '.cookie-notice', '.privacy-banner']
    }
  }
};

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

function normalizeUrl(url) {
  try {
    const urlObj = new URL(url);
    urlObj.hash = ''; // Remove fragments
    return urlObj.toString();
  } catch {
    return null;
  }
}

function isAllowedUrl(url) {
  try {
    const urlObj = new URL(url);

    // Check if domain is allowed
    const isAllowedDomain = config.crawling.allowedDomains.some(domain =>
      urlObj.hostname.includes(domain)
    );

    if (!isAllowedDomain) return false;

    // Check exclude patterns
    const isExcluded = config.crawling.excludePatterns.some(pattern =>
      pattern.test(url)
    );

    return !isExcluded;
  } catch {
    return false;
  }
}

function generateFileNameFromUrl(url) {
  const urlObj = new URL(url);
  const pathParts = urlObj.pathname.split('/').filter(Boolean);
  const fileName = pathParts.length > 0 ? pathParts.join('_') : 'index';
  const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
  return `${fileName}_${hash}`.replace(/[^a-z0-9_-]/gi, '_').substring(0, 100);
}

// Crawler State Management
class CrawlerState {
  constructor(stateFile) {
    this.stateFile = stateFile;
    this.visited = new Set();
    this.toVisit = new Map(); // Changed to Map to store referrer info
    this.failed = new Map();
    this.content = new Map();
    this.notFound = new Set(); // Track 404 pages
    this.headerFooterLinks = new Set(); // Track common navigation links
    this.linkSources = new Map(); // Track where each link was found
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
        console.log(`Loaded state: ${this.visited.size} visited, ${this.toVisit.size} to visit, ${this.notFound.size} 404s`);
      } catch (error) {
        console.error('Error loading state:', error);
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
      lastUpdated: new Date().toISOString()
    };
    fs.writeJSONSync(this.stateFile, state, { spaces: 2 });
  }

  addUrl(url, referrer = null) {
    const normalized = normalizeUrl(url);
    if (normalized && !this.visited.has(normalized) && !this.toVisit.has(normalized) && !this.notFound.has(normalized)) {
      this.toVisit.set(normalized, { referrer, addedAt: new Date().toISOString() });

      // Track where this link came from
      if (referrer) {
        if (!this.linkSources.has(normalized)) {
          this.linkSources.set(normalized, []);
        }
        this.linkSources.get(normalized).push(referrer);
      }
    }
  }

  getNextUrl() {
    const [url, metadata] = this.toVisit.entries().next().value || [null, null];
    if (url) {
      this.toVisit.delete(url);
      return { url, metadata };
    }
    return null;
  }

  markVisited(url) {
    this.visited.add(normalizeUrl(url));
    this.save();
  }

  markFailed(url, error, referrer = null) {
    const normalized = normalizeUrl(url);
    this.failed.set(normalized, {
      error: error.message,
      timestamp: new Date().toISOString(),
      referrer: referrer
    });
    this.save();
  }

  mark404(url, referrer = null) {
    const normalized = normalizeUrl(url);
    this.notFound.add(normalized);
    console.log(`404 found: ${url} (referred by: ${referrer || 'unknown'})`);

    // Log to a separate 404 file for analysis
    const notFoundLog = path.join(config.outputDir, '404_pages.log');
    fs.appendFileSync(notFoundLog, `[${new Date().toISOString()}] 404: ${url} <- ${referrer || 'direct'}\n`);

    this.save();
  }

  markAsHeaderFooterLink(url) {
    const normalized = normalizeUrl(url);
    this.headerFooterLinks.add(normalized);
  }

  isHeaderFooterLink(url) {
    const normalized = normalizeUrl(url);
    return this.headerFooterLinks.has(normalized);
  }

  saveContent(url, content) {
    const normalized = normalizeUrl(url);
    this.content.set(normalized, {
      url: normalized,
      title: content.title,
      text: content.text,
      links: content.links,
      metadata: content.metadata,
      timestamp: new Date().toISOString()
    });
    this.save();
  }
}

// Authentication handler
async function authenticateEPFL(page) {
  console.log('Starting EPFL authentication...');

  const currentUrl = page.url();
  console.log('Current URL:', currentUrl);

  // Check if we're already on a login page
  const isLoginPage = currentUrl.includes('login') ||
                     currentUrl.includes('tequila') ||
                     currentUrl.includes('microsoftonline') ||
                     currentUrl.includes('authentication');

  // If not on a login page, try to find and click login button
  if (!isLoginPage) {
    try {
      // Look for various login buttons/links
      const loginSelectors = [
        '.nav.navbar-nav.login a.pulseit',
        'a[href*="login"]',
        'button:contains("Login")',
        'a:contains("Login")',
        'a:contains("Se connecter")',
        '#login-button',
        '.login-link'
      ];

      let loginClicked = false;
      for (const selector of loginSelectors) {
        try {
          await page.waitForSelector(selector, { timeout: 3000 });
          await page.click(selector);
          loginClicked = true;
          console.log(`Clicked login button: ${selector}`);
          break;
        } catch {
          // Try next selector
        }
      }

      if (!loginClicked) {
        console.log('No login button found, checking if already authenticated or login form present');
      } else {
        await waitRandomTime(config.delays.navigation);
      }
    } catch (error) {
      console.log('No login button found on page');
    }
  }

  // Now handle whichever login form appears
  try {
    // Wait for either Tequila or Microsoft login form
    await page.waitForFunction(
      () => {
        return (
          // Tequila form
          (document.querySelector('#username') && document.querySelector('#password')) ||
          // Microsoft form
          document.querySelector('#i0116') ||
          // Already logged in (check for common logged-in indicators)
          document.querySelector('.user-logged-in') ||
          document.querySelector('[class*="logged"]') ||
          document.body.innerText.includes('Logout') ||
          document.body.innerText.includes('Déconnexion')
        );
      },
      { timeout: 10000 }
    );

    // Check if already logged in
    const isLoggedIn = await page.evaluate(() => {
      return document.querySelector('.user-logged-in') ||
             document.querySelector('[class*="logged"]') ||
             document.body.innerText.includes('Logout') ||
             document.body.innerText.includes('Déconnexion');
    });

    if (isLoggedIn) {
      console.log('Already authenticated');
      return;
    }

    // Handle Tequila authentication
    if (await page.$('#username') && await page.$('#password')) {
      console.log('Detected Tequila login form');

      // Clear fields first (in case of pre-filled values)
      await page.evaluate(() => {
        document.querySelector('#username').value = '';
        document.querySelector('#password').value = '';
      });

      // Type credentials
      await page.type('#username', config.credentials.username, { delay: config.delays.typing });
      await waitRandomTime(500);
      await page.type('#password', config.credentials.password, { delay: config.delays.typing });
      await waitRandomTime(500);

      // Find and click submit button
      const submitSelectors = [
        'input[type="submit"]',
        'button[type="submit"]',
        '#submit',
        'button:contains("Login")',
        'input[value="Login"]',
        'input[value="Se connecter"]'
      ];

      let submitted = false;
      for (const selector of submitSelectors) {
        try {
          await page.click(selector);
          submitted = true;
          console.log(`Clicked submit: ${selector}`);
          break;
        } catch {
          // Try next selector
        }
      }

      if (!submitted) {
        // Try pressing Enter as last resort
        await page.keyboard.press('Enter');
        console.log('Submitted form with Enter key');
      }

      // Wait for navigation after Tequila login
      await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 });
    }

    // Handle Microsoft authentication
    else if (await page.$('#i0116')) {
      console.log('Detected Microsoft login form');

      // Enter username
      await page.type('#i0116', config.credentials.username, { delay: config.delays.typing });
      await waitRandomTime();
      await page.click('#idSIButton9');
      await waitRandomTime(config.delays.navigation);

      // Enter password
      await page.waitForSelector('#i0118', { timeout: 10000 });
      await page.type('#i0118', config.credentials.password, { delay: config.delays.typing });
      await waitRandomTime();
      await page.click('#idSIButton9');
      await waitRandomTime(config.delays.navigation);

      // Handle "Stay signed in" if present
      try {
        await page.waitForSelector('#idSIButton9', { timeout: 5000 });
        await page.click('#idSIButton9');
        await waitRandomTime(config.delays.navigation);
      } catch {
        // No stay signed in prompt
      }
    }

    // Wait for redirect back to EPFL or success indicators
    await page.waitForFunction(
      () => {
        return window.location.href.includes('epfl.ch') &&
               !window.location.href.includes('login') &&
               !window.location.href.includes('tequila') &&
               !window.location.href.includes('authentication');
      },
      { timeout: 60000 }
    );

    console.log('Authentication successful');

  } catch (error) {
    console.error('Authentication failed:', error.message);

    // Take a screenshot for debugging
    try {
      const screenshotPath = path.join(config.outputDir, `auth_error_${Date.now()}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      console.log(`Screenshot saved to: ${screenshotPath}`);
    } catch {}

    throw error;
  }
}

// Expand all collapsed content on the page
async function expandAllCollapsedContent(page) {
  console.log('Checking for collapsed content...');

  // Common selectors for collapsible elements across EPFL sites
  const collapsibleSelectors = [
    // Bootstrap collapse buttons
    'button[data-toggle="collapse"]:not(.collapsed)',
    'button[data-toggle="collapse"].collapsed',
    'a[data-toggle="collapse"]:not(.collapsed)',
    'a[data-toggle="collapse"].collapsed',

    // Accordion elements
    '.accordion-toggle.collapsed',
    '.accordion-button.collapsed',

    // Details/Summary elements
    'details:not([open])',

    // Custom EPFL implementations
    '.collapse-title.collapsed',
    '.expandable-header.collapsed',
    '.toggle-content:not(.expanded)',

    // Other common patterns
    '[aria-expanded="false"]',
    '.show-more-button',
    '.expand-button',
    '.read-more'
  ];

  let expansionCount = 0;
  let previousCount = -1;

  // Keep expanding until no more elements can be expanded
  while (expansionCount !== previousCount) {
    previousCount = expansionCount;

    expansionCount = await page.evaluate((selectors) => {
      let count = 0;

      // Try each selector
      for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);

        elements.forEach(element => {
          try {
            // Skip if already expanded
            const isExpanded = element.getAttribute('aria-expanded') === 'true' ||
                             element.classList.contains('expanded') ||
                             (!element.classList.contains('collapsed') && element.classList.length > 0);

            if (!isExpanded) {
              // Try different methods to expand
              if (element.click) {
                element.click();
                count++;
              } else if (element.dispatchEvent) {
                element.dispatchEvent(new Event('click', { bubbles: true }));
                count++;
              }
            }
          } catch (e) {
            // Silently skip elements that can't be clicked
          }
        });
      }

      // Handle HTML5 details elements
      document.querySelectorAll('details:not([open])').forEach(details => {
        details.open = true;
        count++;
      });

      // Handle elements that toggle visibility
      document.querySelectorAll('[style*="display: none"]').forEach(element => {
        const toggleButton = element.previousElementSibling?.querySelector('button, a') ||
                           element.parentElement?.querySelector('button, a');
        if (toggleButton && toggleButton.onclick) {
          toggleButton.click();
          count++;
        }
      });

      return count;
    }, collapsibleSelectors);

    // Wait for animations and content to load
    if (expansionCount > previousCount) {
      console.log(`Expanded ${expansionCount - previousCount} elements`);
      await waitRandomTime(500); // Wait for animations
      await page.waitForFunction(() => true, { timeout: 500 }).catch(() => {}); // Additional wait for content loading
    }
  }

  console.log(`Total elements expanded: ${expansionCount}`);

  // Final wait to ensure all content is loaded
  if (expansionCount > 0) {
    await waitRandomTime(1000);
  }
}

// Content extraction with header/footer detection
async function extractPageContent(page, currentUrl) {
  return await page.evaluate((selectors, url) => {
    // Remove unwanted elements
    selectors.remove.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => el.remove());
    });

    // Check for 404 indicators
    const is404 =
      document.title.toLowerCase().includes('not found') ||
      document.title.includes('404') ||
      document.body.innerText.toLowerCase().includes('page not found') ||
      document.body.innerText.includes('404') ||
      document.querySelector('.error-404') !== null ||
      document.querySelector('[class*="404"]') !== null ||
      document.body.innerText.toLowerCase().includes("doesn't exist") ||
      document.body.innerText.toLowerCase().includes("removed, or you mistyped");

    if (is404) {
      return { is404: true, title: document.title, url: url };
    }

    // Extract text from main content areas
    let mainContent = '';
    const contentSelectors = [...selectors.main, ...selectors.article];

    for (const selector of contentSelectors) {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        elements.forEach(el => {
          mainContent += el.innerText + '\n\n';
        });
        break;
      }
    }

    // If no main content found, get body text minus header/footer
    if (!mainContent.trim()) {
      // Remove header and footer before extracting
      const headerFooterSelectors = ['header', 'footer', 'nav', '.header', '.footer', '#header', '#footer', '.navbar', '.nav-menu'];
      headerFooterSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => el.remove());
      });
      mainContent = document.body.innerText;
    }

    // Extract all links, categorizing them
    const headerLinks = Array.from(document.querySelectorAll('header a[href], nav a[href], .header a[href], .navbar a[href]')).map(a => ({
      href: a.href,
      text: a.innerText.trim(),
      isNavigation: true
    }));

    const footerLinks = Array.from(document.querySelectorAll('footer a[href], .footer a[href]')).map(a => ({
      href: a.href,
      text: a.innerText.trim(),
      isNavigation: true
    }));

    const contentLinks = Array.from(document.querySelectorAll('main a[href], article a[href], .content a[href], #content a[href]')).map(a => ({
      href: a.href,
      text: a.innerText.trim(),
      isNavigation: false
    }));

    // Get all links if no content links found
    let allLinks = [...headerLinks, ...footerLinks, ...contentLinks];
    if (contentLinks.length === 0) {
      const otherLinks = Array.from(document.querySelectorAll('a[href]'))
        .filter(a => {
          // Filter out already captured navigation links
          const href = a.href;
          return !headerLinks.some(l => l.href === href) && !footerLinks.some(l => l.href === href);
        })
        .map(a => ({
          href: a.href,
          text: a.innerText.trim(),
          isNavigation: false
        }));
      allLinks = [...allLinks, ...otherLinks];
    }

    // Filter out javascript links
    allLinks = allLinks.filter(link => link.href && !link.href.startsWith('javascript:'));

    // Extract metadata
    const metadata = {
      title: document.title,
      description: document.querySelector('meta[name="description"]')?.content || '',
      keywords: document.querySelector('meta[name="keywords"]')?.content || '',
      author: document.querySelector('meta[name="author"]')?.content || '',
      language: document.documentElement.lang || 'en',
      lastModified: document.lastModified
    };

    return {
      title: document.title,
      text: mainContent.trim(),
      links: allLinks,
      metadata: metadata,
      is404: false,
      navigationLinkCount: headerLinks.length + footerLinks.length,
      contentLinkCount: contentLinks.length
    };
  }, config.contentExtraction.selectors, currentUrl);
}

// Save content in multiple formats
async function saveContent(url, content, outputDir) {
  const fileName = generateFileNameFromUrl(url);

  // Save as JSON
  const jsonPath = path.join(outputDir, 'json', `${fileName}.json`);
  fs.ensureDirSync(path.dirname(jsonPath));
  fs.writeJSONSync(jsonPath, {
    url: url,
    crawledAt: new Date().toISOString(),
    ...content
  }, { spaces: 2 });

  // Save as text for RAG
  const textPath = path.join(outputDir, 'text', `${fileName}.txt`);
  fs.ensureDirSync(path.dirname(textPath));
  fs.writeFileSync(textPath, `URL: ${url}\nTitle: ${content.title}\n\n${content.text}`);

  // Save as markdown
  const mdPath = path.join(outputDir, 'markdown', `${fileName}.md`);
  fs.ensureDirSync(path.dirname(mdPath));
  const mdContent = `# ${content.title}\n\nSource: ${url}\n\n${content.text}`;
  fs.writeFileSync(mdPath, mdContent);
}

// PDF handler for documents
// PDF handler for documents
async function downloadPDF(page, url, outputDir) {
  try {
    const fileName = generateFileNameFromUrl(url) + '.pdf';
    const filePath = path.join(outputDir, 'pdfs', fileName);
    fs.ensureDirSync(path.dirname(filePath));

    // Check if URL points to a PDF or other downloadable document
    const isPdfUrl = config.crawling.includeFileTypes.some(ext => url.toLowerCase().includes(ext));

    if (isPdfUrl) {
      console.log(`Downloading PDF directly: ${url}`);

      // Get cookies from the current page session
      const cookies = await page.cookies();

      // Create a new page for downloading
      const downloadPage = await page.browser().newPage();

      try {
        // Set cookies for authentication
        await downloadPage.setCookie(...cookies);

        // Enable request interception to capture the PDF
        await downloadPage.setRequestInterception(true);

        let pdfBuffer = null;

        downloadPage.on('request', request => request.continue());
        downloadPage.on('response', async response => {
          const contentType = response.headers()['content-type'] || '';
          if (contentType.includes('pdf') ||
              contentType.includes('octet-stream') ||
              response.url().toLowerCase().endsWith('.pdf')) {
            try {
              pdfBuffer = await response.buffer();
            } catch (e) {
              console.error('Error getting PDF buffer:', e);
            }
          }
        });

        // Navigate to the PDF URL
        const response = await downloadPage.goto(url, {
          waitUntil: 'networkidle0',
          timeout: 30000
        });

        // If we captured the PDF, save it
        if (pdfBuffer) {
          fs.writeFileSync(filePath, pdfBuffer);
          console.log(`PDF saved successfully: ${fileName}`);
        } else {
          // Fallback: try to get buffer from response
          const responseBuffer = await response.buffer();
          fs.writeFileSync(filePath, responseBuffer);
          console.log(`PDF saved via response buffer: ${fileName}`);
        }

      } finally {
        await downloadPage.close();
      }

    } else {
      // For HTML pages, generate PDF
      console.log(`Generating PDF from HTML page: ${url}`);

      // Wait for page to be fully loaded
      await page.waitForFunction(() => document.readyState === 'complete');
      await waitRandomTime(1000); // Extra wait for dynamic content

      // Set media type to screen for better rendering
      await page.emulateMediaType('screen');

      // Generate PDF
      await page.pdf({
        path: filePath,
        format: 'A4',
        printBackground: true,
        margin: { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' },
        preferCSSPageSize: true,
        displayHeaderFooter: false
      });

      console.log(`PDF generated successfully: ${fileName}`);
    }

    // Verify the PDF was created and has content
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      if (stats.size > 0) {
        return filePath;
      } else {
        console.error(`PDF file is empty: ${fileName}`);
        fs.unlinkSync(filePath);
        return null;
      }
    }

  } catch (error) {
    console.error(`Error handling PDF for ${url}:`, error.message);
    return null;
  }
}

// Main crawler function
async function crawlPage(browser, state, urlInfo, depth = 0) {
  const url = typeof urlInfo === 'string' ? urlInfo : urlInfo.url;
  const referrer = typeof urlInfo === 'object' ? urlInfo.metadata?.referrer : null;

  if (depth > config.crawling.maxDepth || state.visited.size >= config.crawling.maxPages) {
    return;
  }

  const page = await browser.newPage();
  await page.setUserAgent(getRandomUserAgent());
  await page.setDefaultNavigationTimeout(60000);
  await page.setDefaultTimeout(30000);

  try {
    console.log(`Crawling: ${url} (depth: ${depth}, visited: ${state.visited.size})`);
    if (referrer) {
      console.log(`  -> Referred by: ${referrer}`);
    }

    // Navigate to page
    const response = await page.goto(url, { waitUntil: 'networkidle2' });

    // Check if authentication needed
    if (response.url().includes('login') || response.url().includes('microsoftonline')) {
      await authenticateEPFL(page);
      await page.goto(url, { waitUntil: 'networkidle2' });
    }

    await waitRandomTime(config.delays.navigation);

    // Expand all collapsed content before extracting
    await expandAllCollapsedContent(page);

    // Extract content
    const content = await extractPageContent(page, url);

    // Check if it's a 404 page
    if (content.is404) {
      console.log(`404 detected: ${url}`);
      state.mark404(url, referrer);
      return;
    }

    // Save content
    await saveContent(url, content, config.outputDir);
    state.saveContent(url, content);

    // Process links intelligently
    const navigationLinks = content.links.filter(link => link.isNavigation);
    const contentLinks = content.links.filter(link => !link.isNavigation);

    console.log(`  Found ${navigationLinks.length} navigation links, ${contentLinks.length} content links`);

    // Mark navigation links as header/footer links after seeing them a few times
    navigationLinks.forEach(link => {
      const sources = state.linkSources.get(link.href) || [];
      if (sources.length > 3) {
        state.markAsHeaderFooterLink(link.href);
      }
    });

    // Prioritize content links over navigation links
    const linksToProcess = [...contentLinks, ...navigationLinks];

    // Extract and queue new URLs
    for (const link of linksToProcess) {
      if (isAllowedUrl(link.href)) {
        // Skip if it's a known header/footer link and we've already visited it
        if (state.isHeaderFooterLink(link.href) && state.visited.has(normalizeUrl(link.href))) {
          continue;
        }

        state.addUrl(link.href, url);
      }
    }

    // Handle PDFs and documents
    if (config.crawling.includeFileTypes.some(ext => url.includes(ext))) {
      await downloadPDF(page, url, config.outputDir);
    }

    state.markVisited(url);
    console.log(`Successfully crawled: ${url}`);

  } catch (error) {
    console.error(`Error crawling ${url}:`, error.message);
    state.markFailed(url, error, referrer);
  } finally {
    await page.close();
  }
}

// Worker pool for concurrent crawling
class CrawlerPool {
  constructor(browser, state, concurrency) {
    this.browser = browser;
    this.state = state;
    this.concurrency = concurrency;
    this.active = 0;
  }

  async run() {
    while (this.state.toVisit.size > 0 || this.active > 0) {
      while (this.active < this.concurrency && this.state.toVisit.size > 0) {
        const urlInfo = this.state.getNextUrl();
        if (urlInfo) {
          this.active++;
          crawlPage(this.browser, this.state, urlInfo)
            .finally(() => {
              this.active--;
            });
        }
      }
      await setTimeout(1000); // Check every second
    }
  }
}

// Main function
async function main() {
  console.log('Starting EPFL full site scraper for RAG...');
  console.log(`Output directory: ${config.outputDir}`);

  // Create output directories
  fs.ensureDirSync(config.outputDir);
  fs.ensureDirSync(path.join(config.outputDir, 'json'));
  fs.ensureDirSync(path.join(config.outputDir, 'text'));
  fs.ensureDirSync(path.join(config.outputDir, 'markdown'));
  fs.ensureDirSync(path.join(config.outputDir, 'pdfs'));

  // Initialize state
  const stateFile = path.join(config.outputDir, 'crawler_state.json');
  const state = new CrawlerState(stateFile);

  // Add initial URLs if starting fresh
  if (state.toVisit.size === 0 && state.visited.size === 0) {
    config.baseUrls.forEach(url => state.addUrl(url));
  }

  // Initialize log
  const logFile = path.join(config.outputDir, 'crawler.log');
  const log = (message) => {
    const logMessage = `[${new Date().toISOString()}] ${message}\n`;
    console.log(message);
    fs.appendFileSync(logFile, logMessage);
  };

  log('Crawler started');

  // Launch browser
  const browser = await puppeteer.launch({
    headless: true, // Set to false for debugging
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
    // Create crawler pool
    const pool = new CrawlerPool(browser, state, config.crawling.concurrentPages);

    // Start crawling
    await pool.run();

    log(`Crawling completed. Total pages: ${state.visited.size}, Failed: ${state.failed.size}`);

    // Generate summary report
    const report = {
      summary: {
        totalPages: state.visited.size,
        failedPages: state.failed.size,
        notFoundPages: state.notFound.size,
        totalContent: state.content.size,
        headerFooterLinks: state.headerFooterLinks.size,
        crawlStarted: new Date(Math.min(...Array.from(state.content.values()).map(c => new Date(c.timestamp)))),
        crawlEnded: new Date(),
      },
      failedUrls: Array.from(state.failed.entries()),
      notFoundUrls: Array.from(state.notFound),
      topReferrersFor404s: Array.from(state.linkSources.entries())
        .filter(([url]) => state.notFound.has(url))
        .map(([url, sources]) => ({ url, sources }))
        .slice(0, 20),
      contentStats: {
        avgTextLength: Array.from(state.content.values()).reduce((sum, c) => sum + c.text.length, 0) / state.content.size,
        languages: [...new Set(Array.from(state.content.values()).map(c => c.metadata.language))],
        domains: [...new Set(Array.from(state.visited).map(url => new URL(url).hostname))]
      }
    };

    fs.writeJSONSync(path.join(config.outputDir, 'crawl_report.json'), report, { spaces: 2 });
    log('Crawl report generated');

  } catch (error) {
    log(`Fatal error: ${error.message}`);
    console.error(error);
  } finally {
    await browser.close();
  }

  console.log('Scraper finished!');
}

// Execute
if (require.main === module) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

module.exports = { main, CrawlerState, config };