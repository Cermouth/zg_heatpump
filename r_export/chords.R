# Load required libraries
library(circlize)
library(dplyr)
setwd('C:/Users/cancui/OneDrive - ETH Zurich/Dokumente/GitHub/zg_heatpump/r_export')
# Define colors for regions (ETH Color Scheme)
all_regions <- c('CHN', 'JPN', 'KOR', 'ROA', 'EUR', 'DEU', 'ITA', 'AUT', 'CZE', 'ROE', 'USA', 'BRA', 'AUS', 'ROW')

mycolors <- c(
  '#8C0A59',  # CHN - ETH Purple 120%
  '#B73B92',  # JPN - ETH Purple 80%
  '#DC9EC9',  # KOR - ETH Purple 40%
  '#EFD0E3',  # ROA - ETH Purple 20%
  '#215CAF',  # EUR - ETH Blue 100%
  '#08407E',  # DEU - ETH Blue 120%
  '#7A9DCF',  # ITA - ETH Blue 60%
  '#4D7DBF',  # AUT - ETH Blue 80%
  '#D3DEEF',  # CZE - ETH Blue 20%
  '#A6BEDF',  # ROE - ETH Blue 40%
  '#007894',  # USA - ETH Petrol 100%
  '#D48681',  # BRA - ETH Red 60%
  '#8E6713',  # AUS - ETH Bronze 100%
  '#6F6F6F'   # ROW - ETH Grey 100%
)

color_mapping <- setNames(mycolors, all_regions)

# Set output directory
output_dir <- "./"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load all data
all_data <- read.csv("all_trade_flows_2035.csv")

# Ensure From and To are factors with all_regions levels
all_data <- all_data %>%
  mutate(From = factor(From, levels = all_regions),
         To = factor(To, levels = all_regions))

# Define scenarios and technologies
scenarios <- c('Base', 'DemandMet', 'SelfSuff40', 'Tariffs')
technologies <- c('Hp Assembly', 'Hex Manufacturing', 'Compressor Manufacturing')

# Create combined PDF with all scenarios and technologies
pdf(paste0(output_dir, '/_chord_all_scenarios_2035.pdf'), height = 10, width = 10)
par(mfrow = c(4, 3), mar = rep(0.1, 4))

# Loop through scenarios and technologies
for (sce in scenarios) {
  for (tech in technologies) {
    
    # Filter data for this scenario and technology
    data_subset <- all_data %>%
      filter(scenario_name == sce, Product == tech) %>%
      select(From, To, Value) %>%
      unique()
    
    if(nrow(data_subset) > 0) {
      circos.clear()
      circos.par(start.degree = 90, gap.degree = 2, 
                 points.overflow.warning = FALSE)
      
      chordDiagram(data_subset,
                   grid.col = color_mapping,
                   grid.border = NA,
                   transparency = 0.25,
                   directional = 1,
                   direction.type = c("arrows", "diffHeight"),
                   diffHeight = -0.04,
                   annotationTrack = "grid",
                   annotationTrackHeight = c(0.05, 0.1),
                   link.border = 'white',
                   link.lwd = 0.05,
                   link.arr.type = "big.arrow",
                   link.sort = TRUE,
                   link.largest.ontop = TRUE)
      
      # circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
      #   xlim = get.cell.meta.data("xlim")
      #   ylim = get.cell.meta.data("ylim")
      #   sector.name = get.cell.meta.data("sector.index")
      #   circos.text(mean(xlim), ylim[1] + 0.1, sector.name, 
      #               facing = "clockwise", niceFacing = TRUE, 
      #               adj = c(-0.5, 0.5), cex = 0.6)
      # })
      # 
      title(main = paste(sce, tech), cex.main = 0.8)
    }
    
    print(paste("Created:", sce, "-", tech))
  }
}

dev.off()

print("All chord diagrams created successfully in one PDF!")
print(paste("Output:", output_dir, "/chord_all_scenarios_2035.pdf"))