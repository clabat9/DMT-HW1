library(manipulate)


f <-  function(r,b){
x <- seq.int(0,1,length.out = 150)
y <- 1-(1-x^r)^b
plot(x,y,type='l', lty= 1, xaxt='n', yaxt = 'n')
grid(nx = 15, ny = 15, col = "lightgray", lty = 1,
     lwd = par("lwd"), equilogs = TRUE)

axis(1, at=seq(0,1,.02), labels=seq(0,1,.02))
axis(2, at=seq(0,1,.02), labels=seq(0,1,.02))
abline(v = 0.88, lty = 'dashed', col = 'blueviolet')
}

manipulate(f(r,b), r= slider(0, 1000, 5, label = "r"), b= slider(0, 1000, 5, label = "b"))

funzione_da_integrare_FP <- function(x) 1-(1-x^25)^85
funzione_da_integrare_FN <- function(x) (1-x^25)^85

J_fn <- c(0.88, 0.9, 0.95,1)
J_fp <- c(0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)

probabilities_FN <- array(NA, dim = length(J_fn))
probabilities_FP <- array(NA, dim = length(J_fp))

cnt <- 1
for (elem in J_fn){
  probabilities_FN[cnt] <- integrate(funzione_da_integrare_FN, elem, 1)$value
  cnt <- cnt +1
  
}

cnt <- 1
for (elem in J_fp){
  probabilities_FP[cnt] <- integrate(funzione_da_integrare_FP, 0, elem)$value
  cnt <- cnt +1
  
}


ajo <- function(x) 1-(1-x^4)^10

# Create data for the area to shade
cord.x <- c(0,seq(0,1,0.01),1) 
cord.y <- c(0,ajo(seq(0,1,0.01)),1) 

# Make a curve
curve(ajo(x), xlim=c(0,1), ylab = 'p(x)', xaxt='n', lwd = 2) 
grid(nx = 5, ny = 5, col = "lightgray", lty = 1,
     lwd = par("lwd"), equilogs = TRUE)

axis(1, at=seq(0,1,.02), labels=seq(0,1,.02))
abline(v = 0.4755395, lty = 'dashed', col = 'red', lwd = 4)
# Add the shaded area.
polygon(c(0,seq(0,0.4755395,0.01),0.4755395) , c(0,ajo(seq(0,0.4755395,0.01)),0),col='skyblue')

polygon(c(0.4755395,seq(0.4755395,1,0.01),1) , c(1, (ajo(seq(0.4755395,1,0.01))) ,1),col='orchid')

F_F <- function(s){
  accumulate <- 0
  for(p in 0:85){
    accumulate <- accumulate + (((-1)^p) * (choose(85,p)) * (1/(25*p+1)) * (s^(25*p+1)))
  }
  return(s-accumulate)
}
