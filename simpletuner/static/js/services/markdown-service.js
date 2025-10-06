(function initMarkdownService() {
    class MarkdownService {
        constructor() {
            const markdownCtor = window.markdownit || window.markdownIt;
            if (typeof markdownCtor === 'function') {
                this._markdown = markdownCtor({
                    html: true,
                    linkify: true,
                    typographer: true,
                    breaks: false,
                });
            } else {
                this._markdown = null;
            }
        }

        render(readme, assetsList = []) {
            if (!readme) {
                return this._emptyResult();
            }

            if (!this._markdown) {
                return this._fallbackRender(readme);
            }

            const normalized = this._normalizeReadme(readme);
            const frontMatter = this._parseFrontMatter(normalized.frontMatter);
            const assetIndex = this._buildAssetIndex(assetsList);

            const prepared = this._prepareBody(normalized.body);
            const rendered = this._markdown.render(prepared.body);
            const sanitized = this._sanitize(rendered);

            const galleryData = this._buildGallery(frontMatter.data, assetIndex);
            const htmlWithGallery = this._injectGallery(sanitized, prepared.placeholder, galleryData);
            const tags = this._extractTags(frontMatter.data);

            return {
                html: htmlWithGallery,
                tags,
                gallery: galleryData,
                metadata: frontMatter.data,
                frontMatterRaw: frontMatter.raw,
            };
        }

        _emptyResult() {
            return { html: '', tags: [], gallery: [], metadata: {}, frontMatterRaw: null };
        }

        _fallbackRender(readme) {
            const normalized = this._normalizeReadme(readme);
            const escaped = this._escapeHtml(normalized.body || '');
            return {
                html: `<pre>${escaped}</pre>`,
                tags: [],
                gallery: [],
                metadata: {},
                frontMatterRaw: normalized.frontMatter,
            };
        }

        _normalizeReadme(readme) {
            if (typeof readme === 'string') {
                return { frontMatter: null, body: readme };
            }
            const frontMatter = readme?.front_matter ?? readme?.frontMatter ?? null;
            const body = readme?.body ?? '';
            return { frontMatter, body };
        }

        _parseFrontMatter(frontMatter) {
            if (!frontMatter) {
                return { raw: null, data: {} };
            }
            const yamlLib = window.jsyaml || window.jsYAML || window.YAML;
            if (yamlLib && typeof yamlLib.load === 'function') {
                try {
                    const data = yamlLib.load(frontMatter) || {};
                    if (typeof data === 'object' && data !== null) {
                        return { raw: frontMatter, data };
                    }
                } catch (error) {
                    console.warn('Failed to parse README front matter with js-yaml:', error);
                }
            }
            return { raw: frontMatter, data: {} };
        }

        _prepareBody(body) {
            const placeholder = '__MARKDOWN_GALLERY_PLACEHOLDER__';
            const preparedBody = (body || '').replace(/<Gallery\s*\/?>/gi, `\n\n${placeholder}\n\n`);
            return { body: preparedBody, placeholder };
        }

        _sanitize(html) {
            if (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function') {
                return window.DOMPurify.sanitize(html, {
                    USE_PROFILES: { html: true },
                    ALLOW_DATA_ATTR: true,
                });
            }
            return html;
        }

        _buildAssetIndex(assetsList) {
            const index = Object.create(null);
            if (!Array.isArray(assetsList)) {
                return index;
            }
            for (const asset of assetsList) {
                if (!asset || typeof asset !== 'object') {
                    continue;
                }
                const name = asset.name;
                if (typeof name === 'string' && name) {
                    index[name] = asset;
                }
            }
            return index;
        }

        _buildGallery(metadata, assetIndex) {
            const widgets = Array.isArray(metadata?.widget) ? metadata.widget : [];
            const gallery = [];
            widgets.forEach((entry, idx) => {
                if (!entry || typeof entry !== 'object') {
                    return;
                }
                const outputUrl = entry?.output?.url;
                if (typeof outputUrl !== 'string') {
                    return;
                }
                const filename = this._extractAssetName(outputUrl);
                if (!filename) {
                    return;
                }
                const asset = assetIndex[filename];
                if (!asset || !asset.data) {
                    return;
                }
                const parameters = entry.parameters && typeof entry.parameters === 'object' ? entry.parameters : {};
                gallery.push({
                    id: `${filename}-${idx}`,
                    name: filename,
                    title: entry.text || filename,
                    description: entry.description || null,
                    parameters,
                    asset,
                });
            });
            return gallery;
        }

        _extractTags(metadata) {
            const rawTags = metadata?.tags;
            if (!rawTags) {
                return [];
            }
            if (Array.isArray(rawTags)) {
                const result = [];
                rawTags.forEach((tag) => {
                    if (typeof tag === 'string' && tag.trim()) {
                        result.push(tag.trim());
                    }
                });
                return Array.from(new Set(result));
            }
            if (typeof rawTags === 'string') {
                return rawTags.split(',').map((tag) => tag.trim()).filter(Boolean);
            }
            return [];
        }

        _injectGallery(html, placeholder, galleryEntries) {
            if (!galleryEntries.length) {
                return html;
            }

            const container = document.createElement('div');
            container.innerHTML = html;

            const placeholderNodes = this._findPlaceholderNodes(container, placeholder);
            placeholderNodes.forEach((node) => {
                const galleryElement = this._createGalleryElement(galleryEntries);
                if (!galleryElement) {
                    return;
                }

                const parent = node.parentNode;
                if (parent && parent.nodeName === 'P' && parent.textContent.trim() === placeholder) {
                    parent.replaceWith(galleryElement);
                } else {
                    const text = node.nodeValue || '';
                    node.nodeValue = text.replace(placeholder, '').trim();
                    if (parent) {
                        parent.insertBefore(galleryElement, node.nextSibling);
                    }
                }
            });

            return container.innerHTML;
        }

        _findPlaceholderNodes(root, placeholder) {
            const nodes = [];
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
            let current = walker.nextNode();
            while (current) {
                if (current.nodeValue && current.nodeValue.includes(placeholder)) {
                    nodes.push(current);
                }
                current = walker.nextNode();
            }
            return nodes;
        }

        _createGalleryElement(entries) {
            if (!entries.length) {
                return null;
            }
            const wrapper = document.createElement('div');
            wrapper.className = 'checkpoint-gallery readme-gallery';

            entries.forEach((entry) => {
                const asset = entry.asset;
                if (!asset?.data) {
                    return;
                }
                const figure = document.createElement('figure');
                figure.className = 'gallery-item';

                const img = document.createElement('img');
                img.src = asset.data;
                img.alt = entry.title || entry.name;
                figure.appendChild(img);

                const figcaption = document.createElement('figcaption');
                figcaption.className = 'gallery-caption';

                const title = document.createElement('div');
                title.className = 'gallery-caption-title';
                title.textContent = entry.title || entry.name;
                figcaption.appendChild(title);

                const prompt = entry.parameters?.prompt || entry.parameters?.text || null;
                if (prompt && prompt !== entry.title) {
                    const promptLine = document.createElement('div');
                    promptLine.className = 'gallery-caption-prompt';
                    promptLine.textContent = prompt;
                    figcaption.appendChild(promptLine);
                }

                const negative = entry.parameters?.negative_prompt || entry.parameters?.negativePrompt;
                if (negative) {
                    const negativeLine = document.createElement('div');
                    negativeLine.className = 'gallery-caption-negative';
                    negativeLine.textContent = `Negative: ${negative}`;
                    figcaption.appendChild(negativeLine);
                }

                figure.appendChild(figcaption);
                wrapper.appendChild(figure);
            });

            if (!wrapper.childElementCount) {
                return null;
            }
            return wrapper;
        }

        _extractAssetName(url) {
            if (typeof url !== 'string') {
                return null;
            }
            const cleaned = url
                .replace(/^\.\//, '')
                .replace(/^assets\//, '')
                .replace(/^\.\/assets\//, '');
            const parts = cleaned.split('/');
            return parts[parts.length - 1] || null;
        }

        _escapeHtml(value) {
            return String(value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }
    }

    if (!window.markdownService) {
        window.markdownService = new MarkdownService();
    }
})();
