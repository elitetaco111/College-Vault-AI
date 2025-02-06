 _______  _______  _______ _________ _______  _______ _________            _______           _    _________
(  ____ )(  ____ )(  ___  )\__    _/(  ____ \(  ____ \\__   __/  |\     /|(  ___  )|\     /|( \   \__   __/
| (    )|| (    )|| (   ) |   )  (  | (    \/| (    \/   ) (     | )   ( || (   ) || )   ( || (      ) (   
| (____)|| (____)|| |   | |   |  |  | (__    | |         | |     | |   | || (___) || |   | || |      | |   
|  _____)|     __)| |   | |   |  |  |  __)   | |         | |     ( (   ) )|  ___  || |   | || |      | |   
| (      | (\ (   | |   | |   |  |  | (      | |         | |      \ \_/ / | (   ) || |   | || |      | |   
| )      | ) \ \__| (___) ||\_)  )  | (____/\| (____/\   | |       \   /  | )   ( || (___) || (____/\| |   
|/       |/   \__/(_______)(____/   (_______/(_______/   )_(        \_/   |/     \|(_______)(_______/)_(   
                                                                                                           

Deep Learning AI Project for College Vault™ Logo Classification
v0.8.0 - beta

2/6/2025
David Nissly
Kirsten Dusterhoft

This project aims to create a CNN for the purpose of classifying College Vault™ logos
First iteration is going to be a proof of concept for Michigan College Vault™ logos
Upon proof of concept, expand to other College Vault™ schools

Current Version is set up to handle classifying images into 11 different classes
Input requires folder named "Images" (created by scraper)
Output is currently a one-hot encoded class printed to console
Model is saved as model.h5 in the folder

Includes Web Scraper for Image Gathering
To use: include a file in the folder named data.csv which has a single column of skus with the column header "Name"
-Note: column header is case sensitive, Netsuite should export with the column called Name, double check if not working