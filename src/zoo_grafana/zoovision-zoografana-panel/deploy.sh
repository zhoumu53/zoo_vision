npm run build

npx @grafana/sign-plugin@latest --rootUrls https://plumonito.grafana.net,http://localhost:3000

sudo rsync -r dist/* /var/lib/grafana/plugins/zoovision-zoografana-panel/