
To save the file locally on a specific path such as `C:\Users\risha\Downloads`, you need to adjust the file path in the `createExcel` function and ensure the path is correct and writable. Here is the updated code with the specified path:

```javascript
const axios = require('axios');
const ExcelJS = require('exceljs');
const moment = require('moment');
const fs = require('fs');
const path = require('path');

// Shopify API credentials
const SHOP_NAME = '6b3085';
const API_KEY = '45fee06121209911e85213ee8d716aac';
const PASSWORD = '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';
const API_VERSION = '2024-04'; // Update this to the latest version

// Base URL for the Shopify API
const BASE_URL = `https://${API_KEY}:${PASSWORD}@${SHOP_NAME}.myshopify.com/admin/api/${API_VERSION}`;

// Function to fetch products from Shopify API
async function fetchProducts() {
    let products = [];
    let url = `${BASE_URL}/products.json`;

    while (url) {
        try {
            const response = await axios.get(url);
            products = products.concat(response.data.products);

            // Check for pagination
            const linkHeader = response.headers['link'];
            url = null;
            if (linkHeader && linkHeader.includes('rel="next"')) {
                url = linkHeader.split('; rel="next"')[0].slice(1, -1);
            }
        } catch (error) {
            console.error('Error fetching products:', error.message);
            break;
        }
    }

    return products;
}

// Function to create Excel file
async function createExcel(products, filepath) {
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Products');

    worksheet.columns = [
        { header: 'id', key: 'id', width: 20 },
        { header: 'title', key: 'title', width: 30 },
        { header: 'description', key: 'description', width: 50 },
        { header: 'availability', key: 'availability', width: 15 },
        { header: 'condition', key: 'condition', width: 10 },
        { header: 'price', key: 'price', width: 10 },
        { header: 'link', key: 'link', width: 30 },
        { header: 'image_link', key: 'image_link', width: 30 },
        { header: 'brand', key: 'brand', width: 20 },
        { header: 'google_product_category', key: 'google_product_category', width: 20 },
        { header: 'fb_product_category', key: 'fb_product_category', width: 20 },
        { header: 'quantity_to_sell_on_facebook', key: 'quantity_to_sell_on_facebook', width: 15 },
        { header: 'sale_price', key: 'sale_price', width: 10 },
        { header: 'sale_price_effective_date', key: 'sale_price_effective_date', width: 20 },
        { header: 'item_group_id', key: 'item_group_id', width: 20 },
        { header: 'gender', key: 'gender', width: 10 },
        { header: 'color', key: 'color', width: 10 },
        { header: 'size', key: 'size', width: 10 },
        { header: 'age_group', key: 'age_group', width: 10 },
        { header: 'material', key: 'material', width: 10 },
        { header: 'pattern', key: 'pattern', width: 10 },
        { header: 'shipping', key: 'shipping', width: 15 },
        { header: 'shipping_weight', key: 'shipping_weight', width: 15 },
        { header: 'video[0].url', key: 'video_0_url', width: 30 },
        { header: 'video[0].tag[0]', key: 'video_0_tag_0', width: 20 },
        { header: 'style[0]', key: 'style_0', width: 20 }
    ];

    products.forEach(product => {
        const variant = product.variants[0] || {};
        const image = product.images[0] ? product.images[0].src : '';

        worksheet.addRow({
            id: product.id,
            title: product.title,
            description: product.body_html,
            availability: variant.inventory_quantity > 0 ? 'in stock' : 'out of stock',
            condition: 'new',
            price: `${variant.price || '0.00'} USD`,
            link: `https://${SHOP_NAME}.myshopify.com/products/${product.handle}`,
            image_link: image,
            brand: product.vendor || '',
            google_product_category: '',
            fb_product_category: '',
            quantity_to_sell_on_facebook: variant.inventory_quantity || '',
            sale_price: variant.compare_at_price ? `${variant.compare_at_price} USD` : '',
            sale_price_effective_date: '',
            item_group_id: product.id,
            gender: '',
            color: '',
            size: variant.title || '',
            age_group: '',
            material: '',
            pattern: '',
            shipping: '',
            shipping_weight: `${variant.weight || '0'} ${variant.weight_unit || 'kg'}`,
            video_0_url: '',
            video_0_tag_0: '',
            style_0: ''
        });
    });

    await workbook.xlsx.writeFile(filepath);
    return filepath;
}

exports.generateExcel = async (req, res) => {
    try {
        const products = await fetchProducts();
        const filename = `shopify_products_${moment().format('YYYYMMDD_HHmmss')}.xlsx`;
        const filepath = path.join('C:', 'Users', 'risha', 'Downloads', filename);

        console.log('Attempting to create Excel at:', filepath);

        // Ensure the directory exists
        const dir = path.dirname(filepath);
        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir, { recursive: true });
        }

        await createExcel(products, filepath);
        res.json({ success: true, filename: filename });
    } catch (error) {
        console.error('An error occurred:', error);
        res.status(500).json({ success: false, error: error.message });
    }
};

exports.downloadExcel = (req, res) => {
    const filename = req.params.filename;
    const filepath = path.join('C:', 'Users', 'risha', 'Downloads', filename);

    if (fs.existsSync(filepath)) {
        res.download(filepath, filename, (err) => {
            if (err) {
                res.status(500).json({ success: false, error: 'File download failed' });
            }
        });
    } else {
        res.status(404).json({ success: false, error: 'File not found' });
    }
};
```

This code will generate the Excel file and save it to `C:\Users\risha\Downloads`. The `generateExcel` function ensures that the specified directory exists and writes the file to the desired location.