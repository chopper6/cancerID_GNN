# adapted from the original cell router algorithm: 


##### CellRouter Class #####
#install.packages("devtools")
#devtools::install_github('hadley/devtools')
#devtools::install_github("klutometis/roxygen")
#library(roxygen2)

suppressWarnings(suppressMessages(require('reshape')))
suppressWarnings(suppressMessages(require('reshape2')))
suppressWarnings(suppressMessages(require('pheatmap')))
suppressWarnings(suppressMessages(library('scales')))
#library('geomnet')
suppressWarnings(suppressMessages(require('ggplot2')))
suppressWarnings(suppressMessages(require('grid')))

CellRouter <- setClass("CellRouter", slots=
                        c(rawdata="data.frame", ndata="data.frame", scale.data="data.frame",
                          sampTab="data.frame", rdimension="data.frame", pca="list", tsne="list", dc="list", dr.custom="list",
                          var.genes="character",graph="list",signatures="list", sources="character", targets="character",
                          directory="list", paths="data.frame", networks="list",
                          genes.trajectory="character", pathsinfo="list",
                          dynamics="list", clusters="list", correlation="list",
                          top.correlations="list", pathwayenrichment="list"))

#' Check CellRouter object
#' @param object CellRouter object
setValidity("CellRouter",
            function(object){
              msg <- NULL
              if(!is.data.frame(object@rawdata)){
                msg <- c(msg, "expression data must be a data.frame")
              }else if(sum(apply(is.na(object@rawdata), 1, sum) > 0)){
                msn <- c(msg, "expression data must not have NAs")
              }
              if(is.null(msg)){
                TRUE
              }else{
                msg
              }
            })


setGeneric("dotplot", function(object, genes.use, thr,width, height, filename) standardGeneric("dotplot"))
setMethod("dotplot",
          signature = "CellRouter",
          definition = function(object, genes.use, thr, width, height, filename){
            sampTab <- cellrouter@sampTab
            perc <- data.frame(matrix(0, nrow=length(genes.use), ncol=0))
            exp <- perc
            rownames(perc) <- genes.use
            
            for(i in unique(sampTab$population)){
              cells.population <- rownames(sampTab[which(sampTab$population == i),])
              p <- apply(object@ndata[genes.use, cells.population], 1, function(x){sum(x>thr)/length(x)})
              perc <- cbind(perc, p)
              
              v <- apply(object@ndata[genes.use, cells.population], 1, mean)
              exp <- cbind(exp, v)
            }
            colnames(perc) <- unique(sampTab$population)
            colnames(exp) <- unique(sampTab$population)
            rownames(exp) <- sapply(strsplit(rownames(exp), split='__', fixed=TRUE), function(x){x[1]})
            perc$gene <- rownames(perc)
            perc = melt(perc, id.vars  = 'gene')
            exp$gene <- rownames(exp)
            exp$gene <- factor(exp$gene, levels=genes.use)
            exp <- melt(exp, id.vars='gene')
            exp$Percentage <- perc$value*100
            
            pdf(file=filename, width=width, height=height)
            g <- ggplot(exp, aes(gene, variable)) + geom_point(aes(colour=value, size=Percentage)) +
              theme_bw() + xlab("") + ylab("") +
              #theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
              theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
                    panel.grid.minor = element_blank(),
                    panel.grid.major = element_blank(), legend.spacing.y = unit(0, "cm"),
                    panel.border=element_rect(fill = NA, colour=alpha('black', 1),size=1)) +
              #scale_color_gradient("Mean expression",low ="blue", high = "red") + coord_flip()
              scale_colour_gradientn("Mean expression", colours=c("midnightblue","dodgerblue3","goldenrod1","darkorange2")) +
              guides(col=guide_legend(direction="vertical", keywidth = 0.75, keyheight = 0.75, override.aes = list(size=3)))
            print(g)
            dev.off()
            print(g)
          }
          
)

setGeneric("dotplotSignatureScore", function(object, genes.use, width, height, filename) standardGeneric("dotplotSignatureScore"))
setMethod("dotplotSignatureScore",
          signature = "CellRouter",
          definition = function(object, genes.use, width, height, filename){
            sampTab <- object@sampTab
            perc <- data.frame(matrix(0, nrow=length(genes.use), ncol=0))
            exp <- perc
            #rownames(perc) <- genes.use
            
            for(i in unique(sampTab$population)){
              cells.population <- rownames(sampTab[which(sampTab$population == i),])
              ##p <- apply(sampTab[cells.population,], 1, function(x){sum(x>thr)/length(x)})
              #perc <- cbind(perc, p)
              
              v <- apply(sampTab[cells.population, genes.use], 2, mean)
              exp <- cbind(exp, v)
            }
            #colnames(perc) <- unique(sampTab$population)
            colnames(exp) <- unique(sampTab$population)
            rownames(exp) <- sapply(strsplit(rownames(exp), split='__', fixed=TRUE), function(x){x[1]})
            #perc$gene <- rownames(perc)
            #perc = melt(perc, id.vars  = 'gene')
            exp$gene <- rownames(exp)
            exp$gene <- factor(exp$gene, levels=genes.use)
            exp <- melt(exp, id.vars='gene')
            #exp$Percentage <- perc$value*100
            
            pdf(file=filename, width=width, height=height)
            g <- ggplot(exp, aes(gene, variable)) + geom_point(aes(colour=value, size=value)) +
              theme_bw() + xlab("") + ylab("") +
              #theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
              theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
                    panel.grid.minor = element_blank(),
                    panel.grid.major = element_blank(), legend.spacing.y = unit(0, "cm"),
                    panel.border=element_rect(fill = NA, colour=alpha('black', 1),size=1)) +
              #scale_color_gradient("Mean expression",low ="blue", high = "red") + coord_flip()
              scale_colour_gradientn("value", colours=c("midnightblue","dodgerblue3","goldenrod1","darkorange2")) +
              guides(fill= guide_legend(), size=guide_legend(), col=guide_legend(direction="vertical", keywidth = 0.75, keyheight = 0.75, override.aes = list(size=3)))
            print(g)
            dev.off()
            print(g)
          }
          
)


plotEnrichR <- function(enrichments, annotation='Reactome_2016', num.pathways=10, order, width=8, height=8, filename){
  database <- list()
  for(r in names(enrichments)){
    x <- enrichments[[r]][[annotation]][1:num.pathways,]
    x$timepoint <- r
    database[[r]] <- x
  }
  database <- do.call(rbind, database)
  
  df <- database
  df$pvalue <- -log10(df$Adjusted.P.value)
  
  goterms <- as.vector(df$Term) #only top GO terms
  clusters <-unique(df$timepoint)
  
  shared.combined <- data.frame(matrix(NA, nrow=length(clusters), ncol=length(goterms)))
  rownames(shared.combined) <- clusters
  colnames(shared.combined) <- goterms
  
  for(c in clusters){
    for(r in goterms){
      pval <- df[which(as.vector(df$Term) == r & as.vector(df$timepoint) == c), ]
      if(nrow(pval) == 1){
        shared.combined[as.character(pval$timepoint), as.character(pval$Term)] <- pval[, 'pvalue']
      }else{
        shared.combined[as.character(pval$timepoint), as.character(pval$Term)] <- NA
      }
    }
  }
  
  shared.combined$Cluster <- rownames(shared.combined)
  shared.m <- melt(shared.combined, id.vars = c('Cluster'))
  
  colnames(shared.m) <- c('Cluster', 'Description', 'pvalue')
  shared.m$Cluster <- factor(shared.m$Cluster, levels=order)
  shared.m$Description<-gsub("_.*","",as.character(shared.m$Description))
  shared.m$Description <- factor(shared.m$Description, levels=unique(shared.m$Description))
  
  pdf(file=filename, width=width, height=height)
  g <- ggplot(shared.m, aes(Cluster, Description)) + 
    geom_point(aes(colour=pvalue, size=pvalue)) +
    theme_bw() + xlab("") + ylab("") +
    #theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
    theme(axis.text.x=element_text(size=12, angle=45, hjust=1),
          panel.border=element_rect(fill = NA, colour=alpha('black', 1),size=1)) +
    guides(color= guide_legend(title="-log10\n(p-value)"), size=guide_legend(title="-log10\n(p-value)")) +
    #scale_color_gradient("-log10(pvalue)",low ="blue", high = "red")
    scale_colour_gradientn("pvalue", colours=c("midnightblue","dodgerblue3","goldenrod1","darkorange2"))
  print(g)
  dev.off()
}


#' Scale and center the data. Individually regress variables provided in vars.regress using using a linear model. Other models are under development.
#' @param object CellRouter object
#' @param genes.use Vector of genes to scale/center. Default is all genes in object@@ndata
#' @param scale.max Max value in scaled data. Default id 10
#' @param vars.regress Variables to regress out
#'
#' @return CellRouter object
#' @export
setGeneric("scaleData", function(object, genes.use=NULL, scale.max=10, vars.regress=NULL) standardGeneric("scaleData"))
setMethod("scaleData",
          signature="CellRouter",
          definition=function(object, genes.use, scale.max, vars.regress){

            if(is.null(genes.use)){
              #genes.use <- object@var.genes
              genes.use <- rownames(object@ndata)
            }
            data.use <- object@ndata[genes.use,]

            if(!is.null(vars.regress)){
              print('Regression...')
              cat(vars.regress, '\n')
              #data.use <- object@ndata[genes.use, , drop = FALSE];
              #gene.expr <- as.matrix(x = data.use[genes.use, , drop = FALSE])
              gene.expr <- as.matrix(object@ndata[genes.use, , drop = FALSE])
              latent.data <- as.data.frame(object@sampTab[,vars.regress])
              rownames(latent.data) <- rownames(object@sampTab)
              colnames(latent.data) <- vars.regress

              new.data <- sapply(X=genes.use, FUN=function(x){
                regression.mat <- cbind(latent.data, gene.expr[x,])#cbind(latent.data, gene.expr[x,])
                colnames(regression.mat) <- c(colnames(latent.data), "GENE")
                fmla <- as.formula(
                  object = paste0(
                    "GENE ",
                    " ~ ",
                    paste(vars.regress, collapse = "+")
                  )
                )
                lm(formula = fmla, data = regression.mat)$residuals
              })
              data.use <- t(new.data)
            }
            #scale.data <- t(scale(t(object@ndata[genes.use,])))
            scale.data <- t(scale(t(data.use)))
            scale.data[is.na(scale.data)] <- 0
            scale.data[which(scale.data > scale.max)] <- scale.max
            object@scale.data <- as.data.frame(scale.data)

            gc(verbose = FALSE)
            object
          }
)
#' Principal component analysis
#' @param object CelLRouter object
#' @param num.pcs Number of principlal components to compute
#' @param genes.use Genes used for principlam component analysis. Default is all genes in object@@ndata
#' @param seed seed
#' @export

setGeneric("computePCA", function(object, num.pcs, genes.use=NULL, seed=1) standardGeneric("computePCA"))
setMethod("computePCA",
          signature="CellRouter",
          definition=function(object, num.pcs, genes.use, seed){
            #computePCA <- function(data, num.pcs, seed=7){
            library(irlba)
            if (!is.null(seed)) {
              set.seed(seed = seed)
            }
            if(is.null(genes.use)){
              #genes.use <- object@var.genes
              genes.use <- rownames(object@ndata)
            }
            pca <- irlba(A = t(object@scale.data[genes.use,]), nv = num.pcs)
            gene.loadings <- pca$v
            #cell.embeddings <- pca$u
            cell.embeddings <- pca$u %*% diag(pca$d)
            sdev <- pca$d/sqrt(max(1, ncol(data) - 1))
            rownames(gene.loadings) <- rownames(object@scale.data)
            colnames(gene.loadings) <- paste0('PC', 1:num.pcs)
            rownames(cell.embeddings) <- colnames(object@scale.data)
            colnames(cell.embeddings) <- colnames(gene.loadings)

            object@pca <- list(gene.loadings = gene.loadings, cell.embeddings=cell.embeddings, sdev=sdev)
            object@rdimension <- as.data.frame(object@pca$cell.embeddings)

            object
          }
)
#' Perform dimensionality reduction using t-SNE
#' @param object do something
#' @param num.pcs Number of principal components used for dimensionality reduction using t-SNE
#' @param perplexity Perplexity
#' @param max_iter, Max number of iterations
#' @param seed seed
#' @import Rtsne
#' @export
setGeneric("computeTSNE", function(object, num.pcs, perplexity=30, max_iter=2000, seed=7) standardGeneric("computeTSNE"))
setMethod("computeTSNE",
          signature="CellRouter",
          definition=function(object, num.pcs, perplexity, max_iter, seed){

            #computeTSNE <- function(pca, num.pcs, perplexity=40, max_iter=2000, seed=7){
            library(Rtsne)
            if (!is.null(seed)) {
              set.seed(seed = seed)
            }

            #tsne.done <- Rtsne(pca$cell.embeddings[,1:num.pcs], perplexity = perplexity, max_iter = max_iter) #implement this type of tsne analysis in cellrouter, using PCA....
            pca <- object@pca
            #tsne.done <- Rtsne(pca$cell.embeddings[,1:num.pcs], max_iter = max_iter) #implement this type of tsne analysis in cellrouter, using PCA....
            tsne.done <- Rtsne(pca$cell.embeddings[,1:num.pcs], perplexity=perplexity, max_iter = max_iter, check_duplicates = FALSE) #implement this type of tsne analysis in cellrouter, using PCA....
            #tsne.done <- Rtsne(pca$cell.embeddings[,1:num.pcs])
            #plot(tsne.done$Y, xlab= 't-SNE 1',ylab= 't-SNE 2', pch=20)
            m <- tsne.done$Y
            rownames(m) <- rownames(pca$cell.embeddings)
            colnames(m) <- c('tSNE 1', 'tSNE 2')
            object@tsne <- list(cell.embeddings=m)

            object
          }
)
#' Dimensionality reduction using diffusion components
#' @param object CellRouter objecr
#' @param genes.use Genes used for dimensionality reduction
#' @param k Parameter k to be used by the DiffusionMap function from destiny pakcage. default is 20
#' @param sigma Parameter sigma to be used by the DiffusionMap function from destiny pakcage. default is local
#' @param seed seed
#' @export

computeDC <- function(object, genes.use=NULL, k=20,sigma='local', seed=1){
  library(destiny) #anyoing error with DLLs all the time...
  if (!is.null(seed)) {
    set.seed(seed = seed)
  }
  if(is.null(genes.use)){
    genes.use <- object@var.genes
  }

  pca <- object@pca
  diff.comp <- DiffusionMap(as.matrix(t(object@scale.data[genes.use,])), k=20, sigma='local')
  dc <- eigenvectors(diff.comp)
  rownames(dc) <- colnames(object@scale.data)
  object@dc <- list(cell.embeddings=dc)

  object
}

setGeneric("customSpace", function(object, matrix) standardGeneric("customSpace"))
setMethod("customSpace",
          signature="CellRouter",
          definition=function(object, matrix){
            object@dr.custom <- list(cell.embeddings=matrix)
            return(object)
          }
)


#' Normalize the data
#' @param object CellRouter object
#' @export
setGeneric("Normalize", function(object) standardGeneric("Normalize"))
setMethod("Normalize",
          signature="CellRouter",
          definition=function(object){

            x <- object@ndata
            x <- t(t(x)/apply(x,2,sum))
            x <- log1p(x * 10000)
            object@ndata <- as.data.frame(x)
            return(object)
          }
)

#' Identify clusters baed on graph-clustering or model based clustering
#' @param object CellRouter object
#' @param method Method: graph-based clustering or model-based clustering
#' @param k number of nearest neighbors to build a k-nearest neighbors graph
#' @param num.pcs number of principal components that will define the space from where the kNN graph is identified. For example, if num.pcs=10, the kNN graph will be created from a 10-dimensional PCA space
#' @param sim.type Updates the kNN graph to encode cell-cell similarities. Only the Jaccard similarity is implemented in the current version
#' @export

setGeneric("findClusters", function(object, method='graph.clustering', k=20, num.pcs=20, sim.type='jaccard') standardGeneric("findClusters"))
setMethod("findClusters",
          signature = "CellRouter",
          definition = function(object, method, k, num.pcs, sim.type){
            if(method=='graph.clustering'){
              cat('Graph-based clustering\n')
              cat('k: ', k, '\n')
              cat('similarity type: ', sim.type, '\n')
              cat('number of principal components: ', num.pcs, '\n')
              object <- graphClustering(object, k=k, num.pcs=num.pcs, sim.type)
            }else if(method=='model.clustering'){
              cat('Model-based clustering\n')
              cat('number of principal components: ', num.pcs)
              object <- modelClustering(object, num.pcs=num.pcs)
            }
            object
          }
)
#' Model-based clustering using the Mclust package
#' @param object CellRouter object
#' @param num.pcs number of principal components that will define the space used as input to perform model-based clustering
#' @export
setGeneric("modelClustering", function(object, num.pcs) standardGeneric("modelClustering"))
setMethod("modelClustering",
          signature = "CellRouter",
          definition = function(object, num.pcs){
              library(mclust)
              sampTab <- object@sampTab
              colname <- 'population'
              matrix <- object@pca$cell.embeddings[,1:num.pcs]
              mclust <- Mclust(matrix)
              sampTab[names(mclust$classification),colname] <- as.character(mclust$classification)
              sampTab <- sampTab[order(as.numeric(sampTab[[colname]])),]

              colors <- cRampClust(1:length(unique(sampTab[[colname]])), 8)
              names(colors) <- unique(sampTab[[colname]])

              replicate_row <- as.vector(unlist(lapply(split(sampTab, sampTab[[colname]]), nrow)))
              colors_row <- rep(colors, times=replicate_row)
              sampTab[,'colors'] <- colors_row

              object@sampTab <- sampTab
              object
            }
          )

#' Graph-based clustering
#' @param object CellRouter object
#' @param k number of nearest neighbors to build a k-nearest neighbors graph
#' @param num.pcs number of principal components that will define the space from where the kNN graph is identified. For example, if num.pcs=10, the kNN graph will be created from a 10-dimensional PCA space
#' @param sim.type Updates the kNN graph to encode cell-cell similarities. Only the Jaccard similarity is implemented in the current version
#' @param filename Save .gml file containing the kNN graph
#' @export

setGeneric("graphClustering", function(object, k=5, num.pcs, sim.type="jaccard",
                                          filename="graph_subpopulations.gml") standardGeneric("graphClustering"))
setMethod("graphClustering",
          signature = "CellRouter",
          definition = function(object, k, num.pcs, sim.type, filename){
            library('cccd')
            library('proxy') # Library of similarity/dissimilarity measures for 'dist()'
            #matrix <- object@rdimension
            sampTab <- object@sampTab
            matrix <- object@pca$cell.embeddings[,1:num.pcs]

            print('building k-nearest neighbors graph')
            dm <- as.matrix(dist(matrix))
            h <- nng(dx=dm,k=k)
            if(sim.type == 'jaccard'){
              sim <- similarity.jaccard(h, vids=V(h), loops=FALSE)
            }else if(sim.type == 'invlog'){
              sim <- similarity.invlogweighted(h, vids=V(h))
            }
            el <- get.edgelist(h)
            weights <- sim[el]

            E(h)$weight  <- weights
            V(h)$name <- rownames(matrix)
            edges <- as.data.frame(get.edgelist(h))
            rownames(edges) <- paste(edges$V1, edges$V2, sep='_')
            edges$weight <- as.numeric(E(h)$weight)

            rownames(sim) <- V(h)$name
            colnames(sim) <- V(h)$name
            
            ## Community detection to discover subpopulation structure
            print('discoverying subpopulation structure')
            comms <- multilevel.community(as.undirected(h), weights = E(h)$weight)
            V(h)$comms <- membership(comms)
            cell.comms <- commToNames(comms, '') #SP means SubPopulations
            allcells <- as.vector(unlist(cell.comms))

            ## Making sure that color mappings are correct
            sampTab <- sampTab[allcells,] #changesorder of cells in the table
            sampTab$population <- ''
            sampTab$colors <- ''
            comm.colors <- cRampClust(unique(membership(comms)), 8)
            #comm.colors <- cRampClust(unique(membership(comms)), length(unique(membership(comms))))
            names(comm.colors) <- names(cell.comms)
            for(comm in names(cell.comms)){
              sampTab[cell.comms[[comm]], 'population'] <- comm
              sampTab[cell.comms[[comm]], 'colors'] <- comm.colors[comm]
            }
            #sampTab$community <- as.vector(unlist(lapply(strsplit(sampTab$population, split="_"), "[", 2)))
            sampTab$community <- as.vector(sampTab$population)

            ## mapping information to the igraph object
            V(h)[rownames(sampTab)]$subpopulation <- sampTab$colors
            V(h)[rownames(sampTab)]$colors <- sampTab$colors
            V(h)[names(nodeLabels(sampTab,'community'))]$label <- nodeLabels(sampTab, 'community')
            V(h)$size <- 5
            E(h)$arrow.size <- 0.01
            colors <- rainbow(max(membership(comms)))

            #print("plotting graph in RStudio")
            #plot(h,vertex.color=V(h)$colors, vertex.frame.color=V(h)$colors, layout=as.matrix(matrix[,dim.plot]))
            #print('done plotting graph')

            ## Useful information about the graph
            graph <- list()
            graph[['network']] <- h
            graph[['edges']] <- edges
            graph[['similarity_matrix']] <- sim
            graph[['subpopulation']] <- cell.comms
            graph[['communities']] <- comms

            print('updating CellRouter object')
            object@graph <- graph
            if(nrow(object@rawdata) > 0){
             object@rawdata <- object@rawdata[,rownames(sampTab)]
            }
            object@ndata <- object@ndata[,rownames(sampTab)]
            object@sampTab <- sampTab

            write.graph(graph = h, file = filename, format = 'gml')

            rm(h)
            rm(edges)
            rm(sim)

            return(object)
          }
)


#https://briatte.github.io/ggnetwork/

#' Plot kNN graph`
#' @param object CellRouter object
#' @param reduction.type The reduced dimension space used to visualize the kNN graph: tsne, pca, dc or custom
#' @param column.ann column in the metadata table used to annotate the kNN graph. For example, clusters, sorted cell populations
#' @param column.color column in the metadata table corresponding to color used to annotate the kNN graph. Should correspond to the metadata in column.ann
#' @param dims.use dimensions to plot
#' @param width width of output file
#' @param height height og outpur file
#' @param filename name of pdf file generated
#' @import ggplot2
#' @export

setGeneric("plotKNN", function(object, reduction.type="tsne", column.ann, column.color,
                               dims.use=c(1,2), width=10, height=10, filename='kNN_graph.pdf') standardGeneric("plotKNN"))
setMethod("plotKNN",
          signature = "CellRouter",
          definition = function(object, reduction.type, column.ann, column.color,
                                dims.use, width, height, filename){
            library(ggnetwork)
            h <- object@graph$network
            matrix <- slot(object, reduction.type)$cell.embeddings[V(h)$name,dims.use]
            colors <- unique(object@sampTab[[column.color]])
            names(colors) <- unique(as.vector(object@sampTab[[column.ann]]))

            g <- ggnetwork(h, layout=as.matrix(matrix), na.rm=TRUE)
            pdf(file=filename, width=width, height=height)
            g2 <- ggplot(g, aes(x = x, y = y, xend = xend, yend = yend)) +
              geom_edges(alpha=0.3) +
              geom_nodes(aes(color = factor(comms)), size=1) +
              theme_blank() + scale_color_manual("", values=colors) +
              guides(col=guide_legend(direction="vertical", keywidth = 0.75, keyheight = 0.85, override.aes = list(size=3)))
            plot(g2)
            dev.off()
            plot(g2)
          }
)

#' Build kNN graph
#' @param object CellRouter object
#' @param k number of nearest neighbors to build a k-nearest neighbors graph for trajectory reconstruction
#' @param column.ann Column in the metadata table specifying the transitions to be identified. For example, if 'population' is provided, transitions will be identified between clusters previously identified. However, sorted cell populations or customized states can also be used. Check our tutorials for detailed examples.
#' @param num.pcs number of principal components that will define the space from where the kNN graph is identified. For example, if num.pcs=10, the kNN graph will be created from a 10-dimensional PCA space
#' @param sim.type Updates the kNN graph to encode cell-cell similarities. Only the Jaccard similarity is implemented in the current version
#' @param filename name of gml file containing the kNN graph
#' @export

setGeneric("buildKNN", function(object, k=5, column.ann, num.pcs=20, sim.type="jaccard", filename="graph_clusters.gml") standardGeneric("buildKNN"))
setMethod("buildKNN",
          signature = "CellRouter",
          definition = function(object, k, column.ann,  num.pcs, sim.type, filename){

            suppressWarnings(suppressMessages(library('cccd')))
            suppressWarnings(suppressMessages(library('proxy'))) # Library of similarity/dissimilarity measures for 'dist()'
            matrix <- object@pca$cell.embeddings[,1:num.pcs]
            sampTab <- object@sampTab
            smapTab <- sampTab[order(sampTab[[column.ann]]),]

            print('building k-nearest neighbors graph')
            dm <- as.matrix(dist(matrix))
            h <- nng(dx=dm,k=k)
            if(sim.type == 'jaccard'){
              sim <- similarity.jaccard(h, vids=V(h), loops=FALSE)
            }else if(sim.type == 'invlog'){
              sim <- similarity.invlogweighted(h, vids=V(h))
            }
            el <- get.edgelist(h)
            weights <- sim[el]

            E(h)$weight  <- weights
            V(h)$name <- rownames(matrix)
            edges <- as.data.frame(get.edgelist(h))
            rownames(edges) <- paste(edges$V1, edges$V2, sep='_')
            edges$weight <- as.numeric(E(h)$weight)

            rownames(sim) <- V(h)$name
            colnames(sim) <- V(h)$name
            
            V(h)[rownames(sampTab)]$comms <- as.vector(sampTab[[column.ann]]) #cluster->celltype
            cell.comms <- split(sampTab, sampTab[[column.ann]])
            cell.comms <- lapply(cell.comms, rownames)
            #allcells <- as.vector(unlist(cell.comms))
            #sampTab <- sampTab[allcells,] #change order of cells in the table
            #V(h)[names(nodeLabels(sampTab,column.ann))]$label <- nodeLabels(sampTab, column.ann)

            ## Useful information about the graph
            graph <- list()
            graph[['network']] <- h
            graph[['edges']] <- edges
            graph[['similarity_matrix']] <- sim
            graph[['subpopulation']] <- cell.comms
            #graph[['communities']] <- comms

            print('updating CellRouter object')
            object@graph <- graph
            #object@rawdata <- object@rawdata[,rownames(sampTab)]
            object@ndata <- object@ndata[,rownames(sampTab)]
            object@sampTab <- sampTab
            write.graph(graph = h, file = filename, format = 'gml')

            rm(h)
            rm(edges)
            rm(sim)

            return(object)
          }
)
#' Plot heatmap with gene signatures
#' @param object CellRouter object
#' @param markers Genes preferentially expressed in each column.ann. For example, in clusters or sorted populations
#' @param column.ann Column in the metadata table used to annotate the kNN graph. For example, clusters, sorted cell populations
#' @param column.color Color
#' @param num.cells Number of cells to show in the heatmap
#' @param threshold Threshold used to center the data
#' @param genes.show #Vector of gene names to show in the heatmap
#' @param low Color for low expression
#' @param intermediate Color for intermediate expression
#' @param high Color for high expression
#' @param width width
#' @param height height
#' @param filename filename
#' @export

plotSignaturesHeatmap <- function(object, markers, column.ann, column.color, num.cells=NULL, threshold=2, genes.show=NULL,
                                  low='purple', intermediate='black', high='yellow', order=NULL, width, height, filename){
  
  if(is.null(num.cells)){
    print('here')
    cells.keep <- rownames(object@sampTab)
    print(table(object@sampTab[[column.ann]]))
  }else{
    #cells.use <- object@sampTab %>% group_by_(column.ann) %>% sample_n(size = num.cells, replace = TRUE)
    cells.use <- split(object@sampTab, object@sampTab[[column.ann]])
    cells.use <- lapply(cells.use, function(x){
      if(nrow(x) < num.cells){
        cells.use.x <- x[sample(rownames(x), size = nrow(x)),]
      }else{
        cells.use.xx <- x[sample(rownames(x), size = num.cells),]
      }
    })
    cells.use.tmp <- do.call(rbind, cells.use)
    cells.keep <- as.vector(cells.use.tmp$sample_id)
  }

  #data <- object@ndata[,cells.use]
  matrix <- center_with_threshold(object@ndata[,cells.keep], threshold)

  paletteLength <- 100
  #myColor <- colorRampPalette(c("purple","black","yellow"))(paletteLength)
  myColor <- colorRampPalette(c(low, intermediate, high))(paletteLength)
  myBreaks <- c(seq(min(matrix), 0, length.out=ceiling(paletteLength/2) + 1),
                seq(max(matrix)/paletteLength, max(matrix), length.out=floor(paletteLength/2)))


  library(data.table)
  markers2 <- as.data.frame(markers)
  #markers2 <- as.data.frame(markers)
  #markers2 <- as.data.table(markers2)[, .SD[which.max(fc.column)], by=gene]
  #markers2 <- as.data.frame(markers2)
  #markers2 <- as.data.frame(markers)
  #markers2 <- markers2[!duplicated(markers2$gene),] #need to make sure there is no duplicated element...
  sampTab <- object@sampTab
  sampTab <- sampTab[cells.keep,]

  if(column.ann == 'population'){
    markers2 <- markers2[order(as.numeric(markers2$population)),]
    rownames(markers2) <- as.vector(markers2$gene)
    sampTab <- sampTab[order(as.numeric(sampTab$population)),]
  }else if(!is.null(order)){
    markers2 <- markers2[order(factor(markers2$population, levels=order)),]
    sampTab <- sampTab[order(factor(sampTab[[column.ann]], levels=order)),]
  }else{
    markers2 <- markers2[order(as.character(markers2$population)),]
    rownames(markers2) <- as.vector(markers2$gene)
    sampTab <- sampTab[order(as.character(sampTab[[column.ann]])),]
  }
  
  #clusters <- as.vector(object@sampTab$population)
  clusters <- as.vector(sampTab[[column.ann]])
  names(clusters) <- rownames(sampTab)
  #clusters <- clusters[order(clusters)]
  ann_col <- data.frame(population=as.vector(clusters), stringsAsFactors = FALSE)
  rownames(ann_col) <- names(clusters)

  ann_row <- data.frame(signature=as.vector(markers2$population), stringsAsFactors = FALSE)
  rownames(ann_row) <- as.vector(markers2$gene)
  if(!is.null(order)){
    ann_col$population <- factor(ann_col$population, levels=order)
    ann_row$signature <- factor(ann_row$signature, levels=order)
  }

  #colors <- cRampClust(cluster.vector, 8)
  #names(colors) <- cluster.vector
  colors <- unique(sampTab[[column.color]])
  names(colors) <- unique(as.vector(sampTab[[column.ann]]))

  color_lists <- list(population=colors, signature=colors)

  #replicate_row <- as.vector(unlist(lapply(split(ann_row, ann_row$signature), nrow)))
  #colors_row <- rep(colors, times=replicate_row)
  #replicate_col <- as.vector(unlist(lapply(split(ann_col, ann_col$population), nrow)))
  #colors_col <- rep(colors, times=replicate_col)
  index <- getIndexes(ann_col, ann_row, order.columns = unique(ann_col$population), order.rows = unique(ann_row$signature))
  
  if(is.null(genes.show)){
    genes.show <- markers2 %>% group_by(population) %>% top_n(5, fc)
    genes.show <- as.vector(genes.show$gene)
    selected <- as.vector(markers2$gene)
    selected[!(selected %in% genes.show)] <- ""
  }else{
    selected <- as.vector(markers2$gene)
    selected[!(selected %in% genes.show)] <- ""
  }

  pheatmap(matrix[rownames(ann_row),rownames(ann_col)], cluster_rows=FALSE, cluster_cols=FALSE, color = myColor, breaks=myBreaks, fontsize=15,
           gaps_row = index$rowsep, gaps_col = index$colsep, annotation_col = ann_col, annotation_row = ann_row, annotation_colors = color_lists,
           labels_row = selected,
           labels_col = rep("", ncol(matrix)), width=width, height=height, filename=filename)

  #pheatmap(matrix[rownames(ann_row),rownames(ann_col)], cluster_rows=FALSE, cluster_cols=FALSE, color = myColor, breaks=myBreaks,
  #         gaps_row = index$rowsep, gaps_col = index$colsep, annotation_col = ann_col, annotation_row = ann_row, annotation_colors = color_lists,
  #         labels_row = selected,
  #         labels_col = rep("", ncol(matrix)))
  
  gc(verbose = FALSE)
  #pdf(file='results/heatmap_2.pdf', width=10, height=8)
  #heatmap.2(as.matrix(matrix[rownames(ann_row),rownames(ann_col)]), col=myColor,trace="none",
  #          density.info="none", scale="none",margin=c(5,5), key=TRUE, Colv=F, Rowv=F,
  #          srtCol=60, dendrogram="none", cexCol=0.75, cexRow=0.65, labRow=FALSE,
  #          labCol = FALSE, symm=T,symkey=T,
  ##          ColSideColors=colors_col,
  #          RowSideColors=colors_row,
  #          colsep=index$colsep, rowsep=index$rowsep, sepcolor = 'black')
  #dev.off()
}

#' Compute mean expression of each gene in each population
#' @param object CellRouter object
#' @param column Column in the metadata table to group cells for differential expression. For example, if 'population' is specified, population-specific gene signatures will be identified
#' @param pos.only Only uses genes upregulated
#' @param fc.threshold FOld change threshold
#' @export

setGeneric("computeValue", function(object, column='population', fun='max') standardGeneric("computeValue"))
#computeFC(object, column, pos.only, fc.threshold)
setMethod("computeValue",
          signature = "CellRouter",
          definition = function(object, column, fun){
            print('discovering subpopulation-specific gene signatures')
            expDat <- object@ndata
            #membs <- as.vector(object@sampTab$population)
            membs <- as.vector(object@sampTab[[column]])
            diffs <- list()
            for(i in unique(membs)){
              cat('cluster ', i, '\n')
              if(sum(membs == i) == 0) next
              if(fun == 'max'){
                m <- if(sum(membs != i) > 1) apply(expDat[, membs != i], 1, max) else expDat[, membs != i]
                n <- if(sum(membs == i) > 1) apply(expDat[, membs == i], 1, max) else expDat[, membs == i]
              }else if(fun == 'median'){
                m <- if(sum(membs != i) > 1) apply(expDat[, membs != i], 1, median) else expDat[, membs != i]
                n <- if(sum(membs == i) > 1) apply(expDat[, membs == i], 1, median) else expDat[, membs == i]
              }else{
                m <- if(sum(membs != i) > 1) apply(expDat[, membs != i], 1, mean) else expDat[, membs != i]
                n <- if(sum(membs == i) > 1) apply(expDat[, membs == i], 1, mean) else expDat[, membs == i]
              }
              
              d <- data.frame(np=m, p=n) #log scale
              diffs[[i]] <- d
            }
            
            return (diffs)
          }
)



#' Compute fold changes and find gene signatures
#' @param object CellRouter object
#' @param column Column in the metadata table to group cells for differential expression. For example, if 'population' is specified, population-specific gene signatures will be identified
#' @param pos.only Only uses genes upregulated
#' @param fc.threshold FOld change threshold
#' @export

setGeneric("computeFC", function(object, column='population', pos.only=TRUE, fc.threshold=0.25) standardGeneric("computeFC"))
#computeFC(object, column, pos.only, fc.threshold)
setMethod("computeFC",
          signature = "CellRouter",
          definition = function(object, column, pos.only, fc.threshold){
            print('discovering subpopulation-specific gene signatures')
            expDat <- object@ndata
            #membs <- as.vector(object@sampTab$population)
            membs <- as.vector(object@sampTab[[column]])
            diffs <- list()
            for(i in unique(membs)){
              cat('cluster ', i, '\n')
              if(sum(membs == i) == 0) next
              m <- if(sum(membs != i) > 1) apply(expDat[, membs != i], 1, mean) else expDat[, membs != i]
              n <- if(sum(membs == i) > 1) apply(expDat[, membs == i], 1, mean) else expDat[, membs == i]

              #pv <- binompval(m/sum(m),sum(n),n)
              #d <- data.frame(mean.np=m, mean.p=n, fc=n-m, pv=pv) #log scale
              d <- data.frame(mean.np=m, mean.p=n, fc=n-m) #log scale
              if(pos.only){
                genes.use <- rownames(d[which(d$fc > fc.threshold),])
              }else{
                genes.use <- rownames(d[which(abs(d$fc) > fc.threshold),])
              }
              m <- m[genes.use]
              n <- n[genes.use]
              print(length(genes.use))
              #diffs[[i]] <- d
              #d <- data.frame(mean.np=m, mean.p=n, fc=n/m, pv=pv) #linear scale
              #d <- d[!is.infinite(d$fc),]
              #d <- d[which(d$pv < 0.05),]
              #d <- d[order(d$pv, decreasing=FALSE),]
              #d <- d[order(d$mean.p, decreasing=TRUE),]
              #diffs[[i]] <- d[which(d$pv < pvalue & d$fc > foldchange),]

              #for wilcox test
              coldata <- object@sampTab
              coldata[membs == i, "group"] <- "Group1"
              coldata[membs != i, "group"] <- "Group2"

              coldata$group <- factor(x = coldata$group)
              countdata.test <- as.matrix(expDat[genes.use, rownames(x = coldata)])

              p_val <- sapply(X = 1:nrow(x = countdata.test), FUN = function(x) {
                return(wilcox.test(countdata.test[x, ] ~ coldata$group)$p.value)
              })
              
              #to.return <- data.frame(p_val, row.names = rownames(countdata.test))
              d2 <- data.frame(mean.np=m, mean.p=n, fc=n-m, pv=p_val, p.adj=p.adjust(p_val, method='bonferroni'))
              #d2 <- data.frame(mean.np=m, mean.p=n, fc=n-m)
              #diffs[[i]] <- d2[which(d2$pv < pvalue),]
              diffs[[i]] <- d2
            }
            object@signatures <- diffs
            return (object)
          }
)


#' Compute fold changes and find gene signatures
#' @param object CellRouter object
#' @param column Column in the metadata table to group cells for differential expression. For example, if 'population' is specified, population-specific gene signatures will be identified
#' @param pos.only Only uses genes upregulated
#' @param fc.threshold FOld change threshold
#' @export
                                                                      #condition #control
setGeneric("compareTwoGroups", function(object, column='population', group1, group2, fc.threshold=0.25) standardGeneric("compareTwoGroups"))
#computeFC(object, column, pos.only, fc.threshold)
setMethod("compareTwoGroups",
          signature = "CellRouter",
          definition = function(object, column, group1, group2, fc.threshold){
            print('discovering subpopulation-specific gene signatures')
            expDat <- object@ndata
            #membs <- as.vector(object@sampTab$population)
            membs <- as.vector(object@sampTab[[column]])
            diffs <- list()
              m <- if(sum(membs == group2) > 1) apply(expDat[, membs == group2], 1, mean) else expDat[, membs == group2]
              n <- if(sum(membs == group1) > 1) apply(expDat[, membs == group1], 1, mean) else expDat[, membs == group1]
              
              d <- data.frame(mean.np=m, mean.p=n, fc=n-m) #log scale
              genes.use <- rownames(d[which(abs(d$fc) > fc.threshold),])
              
              m <- m[genes.use]
              n <- n[genes.use]
              print(length(genes.use))
              #for wilcox test
              coldata <- object@sampTab
              coldata[membs == group1, "group"] <- "Group1"
              coldata[membs == group2, "group"] <- "Group2"
              
              coldata$group <- factor(x = coldata$group)
              countdata.test <- as.matrix(expDat[genes.use, rownames(x = coldata)])
              
              p_val <- sapply(X = 1:nrow(x = countdata.test), FUN = function(x) {
                return(wilcox.test(countdata.test[x, ] ~ coldata$group)$p.value)
              })
              
              #to.return <- data.frame(p_val, row.names = rownames(countdata.test))
              d2 <- data.frame(mean.np=m, mean.p=n, fc=n-m, pv=p_val, p.adj=p.adjust(p_val, method='bonferroni'))
              #d2 <- data.frame(mean.np=m, mean.p=n, fc=n-m)
              #diffs[[i]] <- d2[which(d2$pv < pvalue),]
              
            
            return (d2)
          }
)


#' Find gene signatures based on a template-matching approach
#' @param expDat Expression matrix
#' @param sampTab Sample annotation table
#' @param qtile qyantile to select top genes correlated with the idealized expression pattern
#' @param remove Remove overlaping genes
#' @param dLevel Groups to compare
#' @export

ranked_findSpecGenes<-function# find genes that are preferentially expressed in specified samples
(expDat, ### expression matrix
 sampTab, ### sample table
 qtile=0.95, ### quantile
 remove=FALSE,
 dLevel="population_name" #### annotation level to group on
){
  cat("Template matching...\n")
  myPatternG<-cn_sampR_to_pattern(as.vector(sampTab[,dLevel]));
  specificSets<-apply(myPatternG, 1, cn_testPattern, expDat=expDat);

  # adaptively extract the best genes per lineage
  cat("First pass identification of specific gene sets...\n")
  cvalT<-vector();
  ctGenes<-list();
  ctNames<-unique(as.vector(sampTab[,dLevel]));
  for(ctName in ctNames){
    x<-specificSets[[ctName]];
    cval<-quantile(x$cval, qtile, na.rm = TRUE);
    tmp<-rownames(x[x$cval>cval,]);
    specificSets[[ctName]] <- specificSets[[ctName]][tmp,]
    ctGenes[[ctName]]<-tmp;
    cvalT<-append(cvalT, cval);
  }
  if(remove){
    cat("Remove common genes...\n");
    # now limit to genes exclusive to each list
    specGenes<-list();
    for(ctName in ctNames){
      others<-setdiff(ctNames, ctName);
      x<-setdiff( ctGenes[[ctName]], unlist(ctGenes[others]));
      specificSets[[ctName]] <- specificSets[[ctName]][x,]
      specificSets[[ctName]]$gene <- rownames(specificSets[[ctName]])
      specGenes[[ctName]]<-x;
    }
    result <- specGenes
  }else {
    result <- ctGenes;
  }
  specificSets <- lapply(specificSets, function(x){x[order(x$cval, decreasing = TRUE),]})
  specificSets <- lapply(specificSets, function(x){colnames(x) <- c('tm.pval', 'cval', 'tm.padj', 'gene'); x})

  specificSets
}
#' Helper function
#' @param sampR sampR
cn_sampR_to_pattern<-function# return a pattern for use in cn_testPattern (template matching)
(sampR){
  d_ids<-unique(as.vector(sampR));
  nnnc<-length(sampR);
  ans<-matrix(nrow=length(d_ids), ncol=nnnc);
  for(i in seq(length(d_ids))){
    x<-rep(0,nnnc);
    x[which(sampR==d_ids[i])]<-1;
    ans[i,]<-x;
  }
  colnames(ans)<-as.vector(sampR);
  rownames(ans)<-d_ids;
  ans;
}
#' Helper function
#' @param pattern pattern
#' @param expDat expression matrix
cn_testPattern<-function(pattern, expDat){
  pval<-vector();
  cval<-vector();
  geneids<-rownames(expDat);
  llfit<-ls.print(lsfit(pattern, t(expDat)), digits=25, print=FALSE);
  xxx<-matrix( unlist(llfit$coef), ncol=8,byrow=TRUE);
  ccorr<-xxx[,6];
  cval<- sqrt(as.numeric(llfit$summary[,2])) * sign(ccorr);
  pval<-as.numeric(xxx[,8]);

  #qval<-qvalue(pval)$qval;
  holm<-p.adjust(pval, method='holm');
  #data.frame(row.names=geneids, pval=pval, cval=cval, qval=qval, holm=holm);
  data.frame(row.names=geneids, pval=pval, cval=cval,holm=holm);
}

#' Identify gene signatures
#' @param object CellRouter object
#' @param column Specify the groups to compare
#' @param test.use Differential expression test to use. Default is wilcox. Alternative is based on template-matching. Possible values: wilcox or template
#' @param pos.only Use only upregulated genes
#' @param fc.threshold Fold change threshold
#' @param fc.tm Wheter to include fold change values in the template-matching differential expression analysis
#' @export

findSignatures <- function(object, column='population', test.use='wilcox', pos.only=TRUE, fc.threshold=0.25, fc.tm=FALSE){
  if(test.use == 'wilcox'){
    cat('Calculating fold changes...', '\n')
    object <- computeFC(object, column, pos.only, fc.threshold)
    markers <- findmarkers(object)
    #markers$gene <- rownames(markers)

  }else if(test.use == 'template'){
    cat('Calculating template-matchings...', '\n')
    signatures <- ranked_findSpecGenes(object@ndata, object@sampTab, qtile=0.99, remove=TRUE, dLevel = column)
    #mylist <- lapply(signatures, functionai(x){x$assignment})
    mylist <- signatures
    for(i in 1:length(mylist) ){ mylist[[i]] <- cbind(mylist[[i]], population=rep(names(mylist[i]), nrow(mylist[[i]])) ) }
    markers <- as.data.frame(do.call(rbind, mylist))
    #rownames(markers) <- as.vector(markers.s$gene)
    rownames(markers) <- as.vector(markers$gene)
    if(fc.tm){
      object <- computeFC(object, column, pos.only, fc.threshold)
      for(signature in names(object@signatures)){
        markers[rownames(signatures[[signature]]), 'log2FC'] <- object@signatures[[signature]][rownames(signatures[[signature]]), 'fc']
        markers[rownames(signatures[[signature]]), 'log2FC_pval'] <- object@signatures[[signature]][rownames(signatures[[signature]]), 'pv']
        markers[rownames(signatures[[signature]]), 'log2FC_p.adj'] <- object@signatures[[signature]][rownames(signatures[[signature]]), 'p.adj']
      }
    }
  }
  markers
}

##Create a table of genes, fc_subpopulation, subpopulation with max expression
setGeneric("findmarkers", function(object) standardGeneric("findmarkers"))
setMethod("findmarkers",
          signature = "CellRouter",
          definition = function(object){
            print('finding subpopulation markers')
            genes <- unique(as.vector(unlist(lapply(object@signatures, rownames))))
            df <- data.frame(matrix(0, nrow=length(genes), ncol=length(object@signatures)))
            rownames(df) <- genes
            colnames(df) <- names(object@signatures)
            for(gene in rownames(df)){
              for(pop in colnames(df)){
                if(gene %in% rownames(object@signatures[[pop]])){
                  df[gene, pop] <- object@signatures[[pop]][gene, 'fc']
                }else{
                  df[gene, pop] <- 0
                }
              }
            }
            x <- apply(df, 1, function(x){names(x)[which(x == max(x))][1]})
            xx <- apply(df, 1, function(x){max(x)})
            df$population <- x
            df$fc <- xx
            #df <- df[order(df$population), c('population', 'fc')]
            df <- df[, c('population', 'fc')]
            for(pop in names(object@signatures)){
              dfx <- df[which(df$population == pop),]
              df[rownames(dfx), 'mean.p'] <- object@signatures[[pop]][rownames(dfx), 'mean.p']
              df[rownames(dfx), 'mean.np'] <- object@signatures[[pop]][rownames(dfx), 'mean.np']
              df[rownames(dfx), 'pval'] <- object@signatures[[pop]][rownames(dfx), 'pv']
              df[rownames(dfx), 'p.adj'] <- object@signatures[[pop]][rownames(dfx), 'p.adj']
            }
            df$gene <- rownames(df)
            df <- df[which(df$fc > 0),] #new line...
            #df <- df[order(nchar(df$population)),]
            return(df)
          }
)


#' Predict a gene regulatory network
#' @param object CellRouter object
#' @param species species
#' @param genes.use genes to include in the gene regulatory network
#' @param zscore zscore threshold to identify putative regulatory interactions
#' @param filename filename of GRN data
#' @export

setGeneric("buildGRN", function(object, species, genes.use=NULL, zscore=5, filename='GRN.R') standardGeneric("buildGRN"))
setMethod("buildGRN",
          signature = "CellRouter",
          definition = function(object, species, genes.use, zscore=5, filename='GRN.R'){
            if(is.null(genes.use)){
              genes.use <- rownames(object@ndata)
            }

            tfs <- find_tfs(species = species)
            grn <- globalGRN(object@ndata[genes.use,], tfs, zscore)
            colnames(grn)[1:2]<-c("TG", "TF");
            ggrn<- ig_tabToIgraph(grn, simplify = TRUE)
            x <- list(GRN=ggrn, GRN_table=grn, tfs=tfs)
            save(x, file=filename)
            
            return(x)
          }
)

#' Predict a gene regulatory network. It allows a customized list of transcription factors or genes (tfs)
#' @param object CellRouter object
#' @param species species
#' @param tfs list of transcription factors to use for GRN reconstruction
#' @param genes.use genes to include in the gene regulatory network
#' @param zscore zscore threshold to identify putative regulatory interactions
#' @param filename filename of GRN data
#' @export

setGeneric("buildGRN2", function(object, species, tfs=NULL, genes.use=NULL, zscore=5, filename='GRN.R') standardGeneric("buildGRN2"))
setMethod("buildGRN2",
          signature = "CellRouter",
          definition = function(object, species, tfs, genes.use, zscore=5, filename='GRN.R'){
            if(is.null(genes.use)){
              genes.use <- rownames(object@ndata)
            }
            if(is.null(tfs)){
              tfs <- find_tfs(species = species)
            }
            grn <- globalGRN(object@ndata[genes.use,], tfs, zscore)
            colnames(grn)[1:2]<-c("TG", "TF");
            ggrn<- ig_tabToIgraph(grn, simplify = TRUE)
            x <- list(GRN=ggrn, GRN_table=grn, tfs=tfs)
            save(x, file=filename)
            
            return(x)
          }
)


#' Plot violin plot
#' @param object CellRouter object
#' @param geneList gene list to plot
#' @param column column to group on
#' @param cols how many clumns in the output figure
#' @param width width
#' @param height height
#' @param filename filename
#' @import ggplot2
#' @export

plotViolin <- function(object, geneList, column, column.color, cols, width=10, height=5, filename, order=NULL, log=TRUE){
  plots <- list()
  sampTab <- object@sampTab
  expDat <- object@ndata
  T0 <- expDat
  allgenes <- data.frame()
  for(g in geneList){
    #cat(time, ' ', dim(T0), '\n')
    genes <- as.data.frame(t(T0[g,]))
    genes$gene <- g
    genes$conditions <- as.vector(sampTab[,column])
    genes.m <- melt(genes, id.var=c('gene',"conditions"))
    allgenes <- rbind(allgenes, genes.m)
  }
  ##log to linear
  if(!log){
    #allgenes$linear <- 2^(allgenes$value) - 1
    allgenes$value <- 2^(allgenes$value) - 1
  }

  colors <- unique(sampTab[[column.color]])
  names(colors) <- unique(sampTab[[column]])
  
  if(is.null(order)){
    order <- unique(allgenes$conditions)
    order <- order[order(as.numeric(order), decreasing=FALSE)]
    allgenes$conditions <- factor(allgenes$conditions, levels=order)
  }else{
    allgenes <- allgenes[order(factor(allgenes$conditions, levels=order)),]
    allgenes$conditions <- factor(allgenes$conditions, levels=order)
  }
  p <- ggplot(allgenes, aes(x=conditions, y=value, fill=conditions)) +
  #p <- ggplot(allgenes, aes(x=conditions, y=linear, fill=conditions)) +
    geom_violin(scale="width") + stat_summary(fun.y=median,geom='point', size=0.5) + #geom_boxplot(alpha=.9) +
    theme_bw() + xlab("") + ylab("") + theme(legend.position="none") +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          #axis.ticks=element_blank(),
          #axis.text.y=element_blank(),
          #axis.ticks.y=element_blank(),
          axis.text.x = element_text(angle = 60, hjust = 1),
          axis.text.y = element_text(size=4),
          strip.text.y = element_text(angle=180),
          panel.border=element_rect(fill = NA, colour=alpha('black', 0.75),size=1),
          strip.background = element_rect(colour="white", fill="white"),
          panel.spacing.x=unit(0.5, "lines"),panel.spacing.y=unit(0, "lines")) + 
    scale_y_continuous(position = "right") +
    scale_fill_manual("", values=colors) +
    facet_wrap(~variable, ncol = cols, strip.position = "left", scales = "free_y") #+ coord_flip()
  #facet_grid(variable ~ ., scales='free') + coord_flip()
  print(p)

  pdf(file=filename, width=width, height=height)
  #multiplot(plotlist = plots, cols=cols)
  print(p)
  dev.off();
  #multiplot(plotlist = plots, cols=cols)
  gc()
}


#' Predict cell cycle phase
#' @param object CellRouter object
#' @param columns columns to be selected from the metadata table
#' @export
predictCellCyle <- function(object, columns){
  cc.scores <- object@sampTab[, columns]
  x <- apply(cc.scores, 1, function(x){
    if(all(x < 0)){
      return('G1')
    }else{
      return(names(x)[which(x == max(x))][1])
    }
  })
  object <- addInfo(object, x, colname = 'Phase')

  object
}

#' Predict state based on scored gene lists
#' @param object CellRouter object
#' @param columns columns to select from the metadata table
#' @param col.name column name to be added to the metadata table after state prediction
#' @export

predictState <- function(object, columns, col.name){
  cc.scores <- object@sampTab[, columns]
  x <- apply(cc.scores, 1, function(x){
    if(all(x < 0)){
      return('Not assigned')
    }else{
      return(names(x)[which(x == max(x))][1])
    }
  })
  object <- addInfo(object, x, colname = col.name)

  object
}

#' Score gene sets
#' @param object CellRouter object
#' @param gene.list gene lists for which the scores will be calculated
#' @param  bins #number of bins to split expression data
#' @param genes.combine default is all genes in object@@ndata
#' @export
scoreGeneSets <- function(object, genes.list, bins=25, genes.combine=NULL){
  if(is.null(genes.combine)){
    genes.combine=rownames(object@ndata)
  }
  ctrl.size <- min(unlist(lapply(genes.list, length)))
  genes.list <- lapply(genes.list, function(x){intersect(x, rownames(object@ndata))})
  cluster.length <- length(x = genes.list)
  data.avg <- Matrix::rowMeans(object@ndata[genes.combine, ])
  data.avg <- data.avg[order(data.avg)]
  data.cut <- as.numeric(x = Hmisc::cut2(
    x = data.avg,
    m = round(length(x = data.avg) / bins)
  ))
  names(data.cut) <- names(data.avg)
  ctrl.use <- vector(mode = "list", length = cluster.length)

  for (i in 1:cluster.length) {
    genes.use <- genes.list[[i]]
    for (j in 1:length(genes.use)) {
      ctrl.use[[i]] <- c(
        ctrl.use[[i]],
        names(sample(data.cut[which(data.cut == data.cut[genes.use[j]])], size = ctrl.size,replace = FALSE))
      )
    }
  }
  ctrl.use <- lapply(ctrl.use, unique)
  ctrl.scores <- matrix(data = numeric(length = 1L),nrow = length(ctrl.use),ncol = ncol(object@ndata))

  for (i in 1:length(ctrl.use)) {
    genes.use <- ctrl.use[[i]]
    ctrl.scores[i, ] <- Matrix::colMeans(object@ndata[genes.use, ])
  }

  genes.scores <- matrix(data = numeric(length = 1L),nrow = cluster.length,ncol = ncol(object@ndata))
  for (i in 1:cluster.length) {
    genes.use <- genes.list[[i]]
    genes.scores[i, ] <- Matrix::colMeans(object@ndata[genes.use, , drop = FALSE])
    #data.use <- object@ndata[genes.use, , drop = FALSE]
    #genes.scores[i, ] <- Matrix::colMeans(x = data.use)
  }
  scores <- genes.scores - ctrl.scores
  rownames(scores) <- names(genes.list)#paste0(col.name, 1:cluster.length)
  scores <- as.data.frame(t(scores))
  rownames(scores) <- colnames(object@ndata)

  #object@sampTab <- cbind(object@sampTab, scores[rownames(object@sampTab),])
  #object@sampTab <- addInfo(object, colname=colnames(scores)) #cbind(object@sampTab, scores[rownames(object@sampTab),])
  for(col.name in colnames(scores)){
    object <- addInfo(object, scores, colname = col.name, metadata.column = col.name)
  }
  object
}


##########################################################

#' Initialize CellRouter object
#' @param .Object object
#' @param rawdata raw data provided as input
#' @param path path from which the raw data will be loaded
#' @param min.genes keep only cells expressing at least min.gene
#' @param min.cells Keep only genes expressed in at least min.cells
#' @param is.expr Threshold to determine whether a gene is expressed or not. By default, genes with raw counts > 0 are considered as expressed.
#' @export

setMethod("initialize",
          signature = "CellRouter",
          definition = function(.Object, rawdata=NULL, path, min.genes=0, min.cells=0, is.expr=0){
            print("Initializing CellRouter object")
            if(is.null(rawdata)){
              rawdata <- as.data.frame(get(load(path)))
            }

            num.genes <- colSums(rawdata > is.expr)
            num.mol <- colSums(rawdata)
            cells.use <- names(num.genes[which(num.genes > min.genes)])
            expdat <- rawdata[, cells.use]
            rawdata <- rawdata[, cells.use]
            genes.use <- rownames(rawdata)
            if (min.cells > 0) {
              num.cells <- rowSums(rawdata > 0)
              genes.use <- names(num.cells[which(num.cells >= min.cells)])
              rawdata <-rawdata[genes.use, ]

            }
            nGene <- num.genes[cells.use]
            nUMI <- num.mol[cells.use]

            .Object@sampTab <- data.frame(sample_id=colnames(rawdata), nGene=nGene, nUMI=nUMI, conditions=colnames(expdat))
            rownames(.Object@sampTab) <- .Object@sampTab$sample_id
            #.Object@rawdata <- rawdata #do not keep raw data matrix to save memory....
            .Object@ndata <- rawdata
            validObject(.Object)
            gc(verbose = FALSE)
            return(.Object)
          }
)

#' Add medata information to CellRouter metadata in object@@sampTab
#' @param object CellRouter object
#' @param metadata Metadata to be added
#' @param colname column name to be added to object@@sampTab in case metadata is a vector
#' @param metadata.column column to selected from the metadata to be added and included in object@@sampTab
#' @export

addInfo <- function(object, metadata, colname, metadata.column='population'){ #uupdate to include data.frames as well...
  sampTab <- object@sampTab
  if(class(metadata) == 'data.frame'){
    sampTab[rownames(metadata), colname] <- as.vector(metadata[[metadata.column]])
  }else{
    sampTab[names(metadata), colname] <- metadata
  }

  colors <- cRampClust(1:length(unique(sampTab[[colname]])), 8) #change $ of colors more properly...
  names(colors) <- unique(sampTab[[colname]])

  replicate_row <- as.vector(unlist(lapply(split(sampTab, sampTab[[colname]]), nrow)))
  colors_row <- rep(colors, times=replicate_row)
  color.column <- paste(colname, 'color',sep='_')
  sampTab[,color.column] <- colors_row

  object@sampTab <- sampTab
  object
}

#' Quality control. Filter out cells based on variables provided in the parameter variables
#' @param object CellRouter object
#' @param variables Filter out cells based on these variables, such as number of detected genes or mitochondrial content
#' @param threshold.low Cells with values lower than the ones provided here are filtered out
#' @param threshold.high Cells with values higher than the ones provided here are filtered out
#' @export

filterCells <- function(object, variables, thresholds.low, thresholds.high){
  sampTab <- object@sampTab

  for(v in 1:length(variables)){
    sampTab <- sampTab[which(as.vector(sampTab[,variables[v]]) < thresholds.high[v] & as.vector(sampTab[,variables[v]]) > thresholds.low[v]),]
  }

  object@rawdata <- object@rawdata[,rownames(sampTab)]
  object@ndata <- object@ndata[,rownames(sampTab)]
  object@sampTab <- sampTab

  object
}


#' Proportion plot.
#' @param object CellRouter object
#' @param condition Column in the metadata table specifying an annotation, such as sorted populations
#' @param population Column in the metddata table specifying another annotation
#' @param width width
#' @param height height
#' @param filename filename
#' @import ggplot2
#' @export
plotProportion <- function(object, condition, population, width, height, filename, order=NULL){
  samples <- object@sampTab
  data2 <- data.frame(cells=samples[[condition]], classification=samples[[population]])
  colors <- as.vector(unique(samples$colors))
  #names(colors) <- unique(as.vector(samples$population))
  names(colors) <- unique(as.vector(samples$population))

  
  if(is.null(order)){
    data2$classification <- factor(data2$classification, levels=unique(as.vector(data2$classification)))
  }else{
    data2 <- data2[order(factor(data2$classification, levels=order)),]
    data2$classification <- factor(data2$classification, levels=order)
  }
  
  
  
  pdf(file=filename, width = width, height=height)
  g <- ggplot(data2,aes(x = classification, fill = cells)) +
    geom_bar(position = "fill", color='black') +
    theme_bw() + theme(legend.position='right',
                       legend.key.size = unit(0.3, "cm"),
                       legend.text=element_text(size=7)) +
    xlab("") + ylab("Proportion") +
    theme(axis.text.x = element_text(size=10, angle=45, hjust=1), axis.title.y = element_text(size = rel(1), angle = 90)) +
    #scale_fill_manual("", values=colors) +
    scale_fill_brewer("", palette = 'Paired') +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0))
  
  #scale_fill_brewer("", palette = 'Paired')
  print(g)
  dev.off()
  print(g)

  data2
}

#' Proportion plot.
#' @param object CellRouter object
#' @param condition Column in the metadata table specifying an annotation, such as sorted populations
#' @param population Column in the metddata table specifying another annotation
#' @param width width
#' @param height height
#' @param filename filename
#' @import ggplot2
#' @export
plotProportion2 <- function(object, condition, population, color.column, width, height, filename, order=NULL){
  samples <- object@sampTab
  data2 <- data.frame(cells=samples[[condition]], classification=samples[[population]])
  colors <- as.vector(unique(samples[[color.column]]))
  #names(colors) <- unique(as.vector(samples$population))
  names(colors) <- unique(as.vector(samples[[condition]]))
  
  
  if(is.null(order)){
    data2$classification <- factor(data2$classification, levels=unique(as.vector(data2$classification)))
  }else{
    data2 <- data2[order(factor(data2$classification, levels=order)),]
    data2$classification <- factor(data2$classification, levels=order)
  }
  
  
  
  pdf(file=filename, width = width, height=height)
  g <- ggplot(data2,aes(x = classification, fill = cells)) +
    geom_bar(position = "fill", color='black') +
    theme_bw() + theme(legend.position='right',
                       legend.key.size = unit(0.4, "cm"),
                       legend.text=element_text(size=7)) +
    xlab("") + ylab("Proportion") +
    theme(axis.text.x = element_text(size=10, angle=45, hjust=1), axis.title.y = element_text(size = rel(1), angle = 90)) +
    scale_fill_manual("", values=colors) +
    #scale_fill_brewer("", palette = 'Paired') +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0))
  
  #scale_fill_brewer("", palette = 'Paired')
  print(g)
  dev.off()
  print(g)
  
}


#' Identify trajectories connecting source and target populations in the kNN graph
#' @param object CellRouter object
#' @param column Column in the metadata table specifying wheter transitions are between clusters or other annotations, such as sorted populations
#' @param libdir Path to Java libraries required
#' @param maindir Directory
#' @param method Method used to determine the source and target cells based on the source and target populations
#' @export
#'
setGeneric("findPaths", function(object, column='population',libdir, maindir, method) standardGeneric("findPaths"))
setMethod("findPaths",
          signature="CellRouter",
          definition=function(object, column, libdir, maindir, method){
            curdir <- getwd()
            dirs <- list()

            if(method %in% c("euclidean"