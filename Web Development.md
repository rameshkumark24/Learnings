# 📘 **WEB DEVELOPMENT — FULL NOTES (Clean Markdown Version)**

*(Everything from your notebook → converted + completed)*

---

# #️⃣ 1. **HTML BASICS**

## ✔ What is HTML?

HTML (HyperText Markup Language) provides the **structure** of a webpage.

---

## ## HTML Document Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Website</title>
</head>
<body>
    <!-- Page content -->
</body>
</html>
```

### ✔ Tags

* Opening tag → `<tagname>`
* Closing tag → `</tagname>`
* Self-closing → `<br />`, `<img />`, `<input />`

---

# #️⃣ 2. **HTML ELEMENTS**

## ✔ Elements with content

```html
<tagname>Content</tagname>
```

## ✔ Attributes

```html
<img src="image.jpg" alt="Description" width="300">
```

---

# #️⃣ 3. **Headings & Paragraphs**

```
<h1> to <h6>
<p>Paragraph</p>
<br>  (line break)
<hr>  (horizontal line)
```

---

# #️⃣ 4. **Text Styling**

| Tag        | Meaning           |
| ---------- | ----------------- |
| `<strong>` | Bold (important)  |
| `<b>`      | Bold (decorative) |
| `<em>`     | Italic (emphasis) |
| `<i>`      | Italic            |
| `<u>`      | Underline         |
| `<mark>`   | Highlight         |
| `<del>`    | Strikethrough     |
| `<ins>`    | Inserted text     |
| `<sub>`    | Subscript         |
| `<sup>`    | Superscript       |
| `<small>`  | Smaller text      |

---

# #️⃣ 5. **Code Elements**

```html
<code>Inline code</code>

<pre>
Preformatted
text
</pre>
```

---

# #️⃣ 6. **Links**

### External link:

```html
<a href="https://example.com">Visit Example</a>
```

### Internal link:

```html
<a href="page.html">Go to Page</a>
```

---

# #️⃣ 7. **Images (Responsive)**

```html
<img src="image.jpg" alt="Description"
     style="max-width: 100%; height: auto;">
```

---

# #️⃣ 8. **Audio & Video**

### Audio:

```html
<audio controls>
    <source src="audio.mp3" type="audio/mpeg">
</audio>
```

### Video:

```html
<video controls width="900">
    <source src="video.mp4" type="video/mp4">
</video>
```

---

# #️⃣ 9. **Lists**

### Unordered list:

```html
<ul>
  <li>Item</li>
</ul>
```

### Ordered list:

```html
<ol>
  <li>Item</li>
</ol>
```

### Description list:

```html
<dl>
  <dt>Term</dt>
  <dd>Description</dd>
</dl>
```

---

# #️⃣ 10. **Tables**

```html
<table>
  <thead>
    <tr><th>Header</th></tr>
  </thead>
  <tbody>
    <tr><td>Data</td></tr>
  </tbody>
</table>
```

---

# #️⃣ 11. **Forms**

```html
<form action="/submit" method="post">
  <fieldset>
      <legend>Info</legend>
      <label>Name:</label>
      <input type="text" name="username">

      <label>Accept?</label>
      <input type="checkbox" value="yes">
  </fieldset>
</form>
```

---

# #️⃣ 12. **CSS BASICS**

CSS = Cascading Style Sheets → controls presentation.

---

## ✔ Selectors

* Element → `p`
* Class → `.className`
* ID → `#idName`
* Attribute → `[attr="value"]`
* Pseudo-class → `a:hover`
* Pseudo-element → `::after`

---

## ✔ Important Properties

### Colors:

* `red`, `blue`
* `#FF0000`
* `rgb(255,0,0)`
* `rgba(255,0,0,0.5)`

### Text:

`font-size`, `font-family`, `text-align`

### Box Model:

* Content
* Padding
* Border
* Margin

---

## ✔ Display & Position

### Display:

`block`, `inline`, `inline-block`, `flex`, `grid`

### Position:

`static`, `relative`, `absolute`, `fixed`, `sticky`

---

## ✔ Flexbox

```css
.container {
  display: flex;
  justify-content: center;
  align-items: center;
}
```

---

## ✔ Grid

```css
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
}
```

---

## ✔ Units

* Absolute → px, pt
* Relative → %, em, rem, vh, vw

---

## ✔ Specificity hierarchy

```
Inline > ID > Class > Element
```

---

# #️⃣ 13. **Website Essentials (Structure)**

✔ Header (logo, menu, navbar)
✔ Hero section (heading, CTA, background)
✔ Main content section
✔ Footer (copyright, social links)
✔ Services / Features
✔ Testimonials
✔ Contact section

---

# #️⃣ 14. **Navigation & Menus**

* Main Menu
* Dropdown Menu
* Mobile Hamburger Menu
* Mega Menu
* Search Bar
* Sticky Header
* Hover animations

---

# #️⃣ 15. **Content & Communication**

* Headline
* Subheadline
* About Section
* Features / Services
* Testimonials
* FAQ
* CTA (Call to Action)
* Contact Form
* Newsletter Signup

---

# #️⃣ 16. **Additional Components**

* Team section
* Timeline / Journey
* Pricing table
* Testimonial carousel
* Animated carousel
* Parallax / Motion scroll
* Background video

---

# #️⃣ 17. **E-Commerce Components**

* Product Listing
* Product Details Page
* Add to Cart
* Cart / Checkout
* Search + Filter
* Wishlist
* Ratings & Reviews
* Related Products
* Discount Banners
* Product hover animations

---

# #️⃣ 18. **SEO & Performance**

* Meta tags (title, description)
* Alt text on images
* Sitemap.xml
* Robots.txt
* Schema markup
* Page speed optimization
* Lazy loading
* Preloaders

---

# #️⃣ 19. **Responsiveness**

* Media queries
* Mobile menu toggle
* Dark mode toggle
* Smooth scrolling
* Scroll animations
* Cursor effects
* Hover & focus animations
* Page transition animations

---

# #️⃣ 20. **Extra / Misc Features**

* Cookie consent banner
* 404 error page
* Login / Signup
* Dashboard / Profile
* Chatbot / Live chat
* WhatsApp Floating button
* Background video
* Confetti / Fireworks effects

---
