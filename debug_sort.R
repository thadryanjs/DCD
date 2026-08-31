library(readxl)
library(dplyr)
df <- read_excel("data/positive-cases-Jennys-data-Edited.xlsx")
print(paste("Class:", class(df$`Date/Time`)[1]))
print(paste("NA count:", sum(is.na(df$`Date/Time`))))
# Check if sorted
is_sorted <- df %>% 
  group_by(Alias) %>% 
  summarise(s = all(diff(`Date/Time`) >= 0, na.rm = TRUE), .groups = "drop") %>% 
  summarise(all_s = all(s)) %>% 
  pull(all_s)
print(paste("Is sorted (na.rm=T):", is_sorted))
