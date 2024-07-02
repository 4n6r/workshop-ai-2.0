const jsdom = require("jsdom");
const fs = require("fs");
const { JSDOM } = jsdom;
const { v4: uuidv4 } = require('uuid');

const urls = fs.readFileSync('./faqs.txt',
    { encoding: 'utf8', flag: 'r' }).split("\n");

console.log(urls)

async function parsePage (url) {
    return await JSDOM.fromURL(url);
}

async function extract (urls) {
    var articles = [];
    for (const url of urls) {
        var res = await parsePage(url);
        res.window.document.querySelectorAll('ing-accordion-item').forEach(item => {
            articles.push({
                id: uuidv4(),
                title: item.getElementsByClassName('headline')[0].innerHTML,
                content: item.getElementsByClassName('accordion-item__content')[0].textContent.replace(/<[^>]*>?/gm, '')
            })
          });
    }

    return articles;
}

(async () => {
    var articles = await extract(urls);
    fs.writeFileSync("./articles.json", JSON.stringify(articles))
    console.log(articles);
  })();